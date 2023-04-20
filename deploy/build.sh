#!/bin/bash

mkdir output
cd deploy && make build
cp bin/manager ../output
