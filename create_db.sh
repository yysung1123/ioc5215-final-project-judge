#!/bin/bash

rm -rf sqlite.db && python3 -c 'from judge import init_db; init_db()'
