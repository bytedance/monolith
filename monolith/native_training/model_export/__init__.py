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

import sys as _sys
import monolith.native_training.model_export.export_context as export_context
import monolith.native_training.model_export.saved_model_exporters as saved_model_exporters

_sys.modules['monolith.model_export.export_context'] = export_context
_sys.modules[
    'monolith.model_export.saved_model_exporters'] = saved_model_exporters
del _sys
