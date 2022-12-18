# Copyright 2022 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import threading
import time
import shutil
import logging
from monolith.native_training.metric import cli

#from absl import logging


# copy latest model from source path(p2p path) to receive path(model path)
class ModelManager(object):

  WRITE_DONE = '.write.done'
  READ_LOCK = '.read.lock'

  def __init__(self, model_name, source_path, receive_path, use_metrics):
    self._worker = None
    self._source_path = source_path
    self._receive_path = receive_path
    self._model_name = model_name
    self._models = {}  # model_name to version list
    self._latest_models = {}
    self._wait_timeout = 1200
    self._loop_thread = None
    self._loop_interval = 30
    self._exist = False
    self._remain_version_num = 5
    self._lock_files = set()
    self._use_metrics = use_metrics
    self._metrics = None

    if self._model_name and self._use_metrics:
      self.init_metrics()

  def init_metrics(self):
    self._metrics = cli.get_cli('data.monolith_serving.online')

  def stop(self):
    self._exist = True
    if self._loop_thread:
      self._loop_thread.join()

  def start(self):
    ret = False
    try:
      ret = self._start()
    except Exception as e:
      logging.error('model manager start failed: %s', str(e))
      ret = False
    return ret

  def _start(self):
    if self._model_name is None:
      logging.info('ModelManager is not needed')
      return True

    # delete receive path first
    if not self.delete(self._receive_path):
      return False

    # wait for the source path
    if not self.wait_for_download():
      return False

    # do loop once to copy model
    while True:
      try:
        if self.loop_once():
          break
      except BaseException as err:
        logging.error('model manager loop once failed: %s', str(err))

      logging.info('loop once failed, wait for ready model')
      time.sleep(10)

    self.remove_read_lock()
    self._loop_thread = threading.Thread(target=self.run,
                                         name="thread-model_manager")
    self._loop_thread.start()

    return True

  def touch(self, file):
    try:
      f = open(file, 'w+')
      f.close()
      return True
    except BaseException:
      pass
    return False

  def run(self):
    while not self._exist:
      try:
        ret = self.loop_once()
        self.remove_read_lock()
        if not ret:
          logging.error('model manager loop once failed')
      except BaseException as err:
        logging.error('model manager loop once failed: %s', str(err))

      if self._use_metrics:
        self.check_model_update_time()
      time.sleep(self._loop_interval)
      self.remove_old_file()

  def check_model_update_time(self):
    if not self._metrics:
      return

    if self._model_name not in self._latest_models:
      logging.error('model %s not in _latest_models: %s', self._model_name,
                    str(self._latest_models))
      self._metrics.emit_counter('loop_once_failed',
                                 1,
                                 tagkv={'model': self._model_name})
      return
    version, update_time = self._latest_models[self._model_name]
    cur_time = int(time.time())
    self._metrics.emit_store('version.delay',
                             cur_time - int(version),
                             tagkv={'model': self._model_name})
    self._metrics.emit_store('update.delay',
                             cur_time - update_time,
                             tagkv={'model': self._model_name})

  def remove_old_file(self):
    for model_name in self._models:
      model_files_list = self._models[model_name]
      if len(model_files_list) > self._remain_version_num:
        old_files = model_files_list.pop(0)
        for old_file in old_files[1]:
          self.delete(old_file)

  def create_read_lock(self, name):
    lock_name = name + self.READ_LOCK
    if self.touch(lock_name):
      return lock_name
    else:
      logging.error("create lock %s failed", lock_name)
      return lock_name

  def remove_read_lock(self):
    for lock_file in self._lock_files:
      self.delete(lock_file)
    self._lock_files.clear()

    # remove other lock
    ret = list(os.walk(self._source_path))
    if len(ret) == 0:
      return

    root, dirs, files = ret[0]

    for file in files:
      if file.endswith(self.READ_LOCK):
        completed_name = os.join(root, file)
        logging.info('delete lock file: %s', completed_name)
        self.delete(completed_name)

  def loop_once(self):
    source_data = {}
    result = True
    try:
      source_data = self.get_source_data()
    except BaseException as err:
      logging.error('get download data failed: %s', str(err))
      return False
    for model_name in source_data:
      new_version = source_data[model_name][0]
      if model_name in self._models and len(self._models[model_name]) > 0:
        old_version = self._models[model_name][-1][0]
        if old_version >= new_version:
          continue

      ret, file_list = self.copy_model(model_name, new_version,
                                       source_data[model_name][1])
      if ret:
        if model_name not in self._models:
          self._models[model_name] = []

        self._models[model_name].append((new_version, file_list))
        cur_time = int(time.time())
        self._latest_models[model_name] = (new_version, cur_time)
        logging.info(f'{model_name} update to {new_version}')
      else:
        logging.error(f'copy {model_name} failed')
        result = False

    return result

  def copy_model(self, model_name, version, model_data):
    sub_model_num = len(model_data)
    ready_data = []
    result = []
    ready_num = 0
    for sub_model_name, sub_model_data in model_data:
      # sub_model_name: ps_0/version
      # sub_model_data: /xxx/model_name@version/ps_0/version
      try:
        src_file = sub_model_data
        dst_file = os.path.join(self._receive_path, model_name, sub_model_name)
        temp_dst_file = dst_file + '-temp'

        result.append(dst_file)

        if os.path.exists(dst_file):
          logging.error(f'{dst_file} exist')
          ready_num += 1
          continue

        if os.path.exists(temp_dst_file):
          logging.error(f'{temp_dst_file} exist')
          ready_num += 1
          ready_data.append((temp_dst_file, dst_file))
          continue

        shutil.copytree(src_file, temp_dst_file)
        ready_data.append((temp_dst_file, dst_file))
        ready_num += 1
      except BaseException as err:
        logging.error('copy model %s -> %s faild: %s', src_file, temp_dst_file,
                      str(err))
        self.delete(temp_dst_file)
        break

    if ready_num != sub_model_num:
      logging.error(
          f'copy model faild, ready_num={ready_num}, expect_num={sub_model_num}'
      )
      for data in ready_data:
        self.delete(data[0])
      return False, []

    for data in ready_data:
      os.rename(data[0], data[1])

    return True, result

  def wait_for_download(self):
    duartion = 0
    download_path_ready = os.path.exists(self._source_path)

    while not download_path_ready and duartion < self._wait_timeout:
      logging.info(f'wait {self._source_path} created')
      time.sleep(10)
      duartion += 10
      download_path_ready = os.path.exists(self._source_path)

    if not download_path_ready:
      logging.error(f'{self._source_path} is not ready')
      return False

    while duartion < self._wait_timeout:
      ret = list(os.walk(self._source_path))

      if len(ret) > 0:
        root, dirs, files = ret[0]

        for file in files:
          if file.endswith(self.WRITE_DONE) and file.startswith(
              self._model_name):
            logging.info(f'{file} is ready')
            return True

      logging.info('no ready model found')
      time.sleep(10)
      duartion += 10
    logging.error('no ready model found')
    return False

  def get_source_data(self):
    source_data = {}
    ret = list(os.walk(self._source_path))
    if len(ret) == 0:
      logging.error(f'{self._source_path} is empty')
      return source_data

    root, dirs, files = ret[0]

    done_file_set = set()
    for file in files:
      if file.endswith(self.WRITE_DONE) and file.startswith(self._model_name):
        done_file_set.add(file)

    for model_data in dirs:
      lock_file = self.create_read_lock(os.path.join(root, model_data))
      self._lock_files.add(lock_file)

      if self.get_done_file(model_data) in done_file_set:
        data = model_data.split('@')
        if len(data) != 2:
          logging.error(f'{model_data} is not valid')
          continue

        model_name, version = data

        # real_path: /xxx/model_name@version/model_name
        real_path = os.path.join(root, model_data, model_name)
        # version_data: [(ps_0/version,/xxx/model_name@version/ps_0/version), (..,..)]
        version_data = self.get_version_data(real_path, version)

        if len(version_data) == 0:
          continue

        if model_name not in source_data:
          source_data[model_name] = (version, version_data, real_path)
        else:
          old_data = source_data[model_name]
          if old_data[0] < version:
            source_data[model_name] = (version, version_data, real_path)

    return source_data

  def get_version_data(self, path, version):
    ret = list(os.walk(path))
    if len(ret) == 0:
      logging.error(f'get version data [{path}] failed')
      return []

    sub_root, sub_dirs, sub_files = ret[0]
    if len(sub_dirs) == 0:
      return []

    res = []
    for sub_dir in sub_dirs:
      # sub_dir: ps_0
      # version_dir: /xxx/model_name@version/ps_0/version
      version_dir = os.path.join(sub_root, sub_dir, version)
      if not os.path.exists(version_dir):
        logging.error(f'{version_dir} not exist')
        return []
      else:
        res.append((os.path.join(sub_dir, version), version_dir))

    return res

  def get_done_file(self, file):
    return file + self.WRITE_DONE

  def delete(self, file):
    try:
      if not os.path.exists(file):
        return True

      if os.path.isfile(file):
        os.remove(file)
      else:
        shutil.rmtree(file)
      return True
    except BaseException as err:
      logging.error('delete [%s] failed: %s', file, str(err))
    return False


