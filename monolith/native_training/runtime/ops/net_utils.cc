// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <cstdio>
#include <cstring>
#include <map>

namespace tensorflow {
namespace monolith_tf {

std::multimap<std::string, std::string> GetLocalIpAddreeses() {
  std::multimap<std::string, std::string> addresses;
  ifaddrs *ifaddrs_list = nullptr;
  getifaddrs(&ifaddrs_list);

  for (ifaddrs *ifa = ifaddrs_list; ifa != nullptr; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr) {
      continue;
    }
    void *tmp_addr = nullptr;
    if (ifa->ifa_addr->sa_family == AF_INET) {  // check it is IP4
      tmp_addr = &(reinterpret_cast<sockaddr_in *>(ifa->ifa_addr)->sin_addr);
      char buffer[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmp_addr, buffer, INET_ADDRSTRLEN);
      addresses.insert({ifa->ifa_name, buffer});
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {  // check it is IP6
      // is a valid IP6 Address
      tmp_addr = &(reinterpret_cast<sockaddr_in6 *>(ifa->ifa_addr)->sin6_addr);
      char buffer[INET6_ADDRSTRLEN];
      inet_ntop(AF_INET6, tmp_addr, buffer, INET6_ADDRSTRLEN);
      addresses.insert({ifa->ifa_name, buffer});
    }
  }
  if (ifaddrs_list != nullptr) freeifaddrs(ifaddrs_list);
  return addresses;
}

std::string GetMyHostIp() {
  // If we are in TCE, env var will provide ip to us.
  char *ip = getenv("MY_HOST_IP");
  if (ip != nullptr) {
    return ip;
  }
  auto addresses = GetLocalIpAddreeses();
  auto it = addresses.find("eth0");
  if (it == addresses.end()) {
    return "";
  }
  return it->second;
}

}  // namespace monolith_tf
}  // namespace tensorflow
