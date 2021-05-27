#!/bin/bash

echo > sqlite.db
python3 -c 'from judge import init_db; init_db()'
