#!/bin/bash
nvcc --relocatable-device-code=true main.cu virtual_memory.cu user_program.cu -o main.out
