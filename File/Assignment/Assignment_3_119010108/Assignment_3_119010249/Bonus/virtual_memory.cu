#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = -1;// at fisrt 1-1024 page is in the memory
    vm->invert_page_table[i + vm->PAGE_ENTRIES*2] = -1;      // pid initialized to be -1
  }
}

__device__ int* LRU_table;

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;                  /* page size = 32b */
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;    /* 16KB for page table setting */
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;  /* 32KB for physical memory */
  vm->STORAGE_SIZE = STORAGE_SIZE;          /* 128KB for the disk storage */
  vm->PAGE_ENTRIES = PAGE_ENTRIES;          /* = PHYSICAL_MEM_SIZE / PAGE_SIZE =  1024*/

  LRU_table = (int*)malloc(PAGE_ENTRIES*sizeof(int));
  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  int base = addr / (1<<5);
  int offset = addr % (1<<5);
  int physical_addr_base;
  int physical_addr;
  int found = 0;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {      /* search for the page number in the page table */
    if (vm->invert_page_table[i+vm->PAGE_ENTRIES] == base && vm->invert_page_table[i+vm->PAGE_ENTRIES*2] == threadIdx.x){     /* if founded, the index is the base */
        physical_addr_base = i;
        vm->invert_page_table[i] = 0;               /* if use it set the valid bit to be valid */
        found = 1;        /* means this page is now in the main memory */
        for (int j = 0; j < vm->PAGE_ENTRIES; j++){
          LRU_table[j] -= 1;
        }
        LRU_table[i] = 0;
        break;
    }
  }
  physical_addr = physical_addr_base*(1<<5)+offset;
  if (found){
    uchar back_value = vm->buffer[physical_addr];
    return back_value;              /* return the corresponding value */
  }
  else{    /* it means the page fault, then do the swap */
    *vm->pagefault_num_ptr += 1;
    int quit_index=0;
    int min = -100;
    for (int i = 0; i < vm->PAGE_ENTRIES; i++){
      if (LRU_table[i] < min){
        quit_index = i;
        min = LRU_table[i];
      }
      if (vm->invert_page_table[i] == 0x80000000){   /* we firstly use the invlalid(empty) page */
          quit_index = i;
          vm->invert_page_table[i] = 0;               
          break;
      }
    }
    /* here put the quit one to the backstore */
    int page_number = vm->invert_page_table[quit_index+vm->PAGE_ENTRIES];
    int back_addr_base = page_number*(1<<5);
    for (int i = back_addr_base; i < vm->PAGESIZE+back_addr_base; i++){
      if (i >= 0){  
        vm->storage[i] = vm->buffer[quit_index*(1<<5)+i-back_addr_base];
      }
      vm->buffer[quit_index*(1<<5)+i-back_addr_base] = NULL;
      vm->buffer[quit_index*(1<<5)+i-back_addr_base] = vm->storage[base*(1<<5)+i-back_addr_base]; /* load the subsititue page into buffer */
    }
    /* now put the new one into the page table and LRU table*/
    vm->invert_page_table[quit_index+vm->PAGE_ENTRIES] = base;
    vm->invert_page_table[quit_index+vm->PAGE_ENTRIES*2] = threadIdx.x;
    LRU_table[quit_index] = 0;  /* this one can be changed to other number */
    /* do the write again */
    return vm_read(vm, addr);
  }
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  int base = addr / (1<<5);      
  int offset = addr % (1<<5);           /* get the last 5 bits */
  int physical_addr_base;
  int physical_addr;
  int found = 0;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {      /* search for the page number in the page table */
    if (vm->invert_page_table[i+vm->PAGE_ENTRIES] == base && vm->invert_page_table[i+vm->PAGE_ENTRIES*2] == threadIdx.x){     /* if founded, the index is the base */
        physical_addr_base = i;
        found = 1;        /* means this page is now in the main memory */
        for (int j = 0; j < vm->PAGE_ENTRIES; j++){
          LRU_table[j] -= 1;
        }
        LRU_table[i] = 0;
        break;
    }
  }
  physical_addr = physical_addr_base*(1<<5)+offset;
  if (found){
    vm->buffer[physical_addr] = value;              /* write the value to this physical space */
  }
  else{    /* it means the page fault, then do the swap */
    *vm->pagefault_num_ptr += 1;
    int quit_index=0;
    int min = -100;
    for (int i = 0; i < vm->PAGE_ENTRIES; i++){     /* decide who is the least recently used */
      if (LRU_table[i] < min){
        quit_index = i;
        min = LRU_table[i];
      }
      if (vm->invert_page_table[i] == 0x80000000){   /* we firstly use the invlalid(empty) page */
          quit_index = i;
          vm->invert_page_table[i] = 0;               
          break;
      }
    }
    /* here put the quit one to the backstore */
    int page_number = vm->invert_page_table[quit_index+vm->PAGE_ENTRIES];
    int back_addr_base = page_number*(1<<5);
    for (int i = back_addr_base; i < vm->PAGESIZE+back_addr_base; i++){
      if (i >= 0){                                                         /* < 0 means it is empty, then, no need to swap it out */
        vm->storage[i] = vm->buffer[quit_index*(1<<5)+i-back_addr_base];   /* write the previous one back */
      }
      vm->buffer[quit_index*(1<<5)+i-back_addr_base] = NULL;             /* clear previous position */
      vm->buffer[quit_index*(1<<5)+i-back_addr_base] = vm->storage[base*(1<<5)+i-back_addr_base]; /* load the subsititue page into buffer */
    }
    /* now put the new one into the page table and LRU table*/
    vm->invert_page_table[quit_index+vm->PAGE_ENTRIES] = base;
    vm->invert_page_table[quit_index+vm->PAGE_ENTRIES*2] = threadIdx.x;      /* remark the thread id */
    LRU_table[quit_index] = 0;  /* this one can be changed to other number */
    /* do the write again */
    vm_write(vm, addr, value);
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int addr = 0; addr < input_size; addr++){
    results[addr] = vm_read(vm, addr);
  }
}

