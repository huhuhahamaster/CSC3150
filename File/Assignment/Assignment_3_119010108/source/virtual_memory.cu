#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ bool ISPageFault(VirtualMemory *vm, u32 page_num) {

	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == page_num) {
			return false;
		}
	}

	(*vm->pagefault_num_ptr)++;
	return true;
}

__device__ void SwapToStorage(VirtualMemory *vm, u32 frame_num) {

	/* find the page in corresponding frame and load it into disk */
	u32 page_num = vm->invert_page_table[frame_num];

	for (int i = 0; i < 32; i++) {
		vm->storage[page_num * 32 + i] = vm->buffer[frame_num * 32 + i];
	}
}

__device__ void ISFrameFull(VirtualMemory *vm, u32 page_num, u32 frame_num) {

	if (vm->invert_page_table[frame_num] != 0x80000000) {
		SwapToStorage(vm, frame_num);
	}
}

__device__ int FindFrameIndex(VirtualMemory *vm, u32 frame_num) {

	/* find the index of the given frame num */
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == frame_num) return i;
	}

	printf("The index is out of index\n");
	return -1;
}

__device__ void UpdateTable(VirtualMemory *vm, u32 frame_num) {

  	int index = FindFrameIndex(vm, frame_num);
	
	/* store the value for current  index */
	int temp = vm->invert_page_table[vm->PAGE_ENTRIES + index];

	/* move one forward for another index after current index */
	for (int i = index; i < vm->PAGE_ENTRIES - 1; i++) {
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = vm->invert_page_table[i + vm->PAGE_ENTRIES + 1];
	}

	/* move the current value to the last, represent most frequent use */
	vm->invert_page_table[2 * vm->PAGE_ENTRIES - 1] = temp;
}

__device__ int FindFrameNum(VirtualMemory *vm,u32 page_num) {

	/* find the frame num given the page num */
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == page_num) return i;
	}
	printf("frame num is out of index\n");
	return -1;
}

__device__ void SwapToMemory(VirtualMemory *vm, u32 frame_num, u32 page_num) {

	/* current page num needed to be swaped */
	u32 original_page_num = vm->invert_page_table[frame_num];
	for (int i = 0; i < 32; i++) {
		/* move the current page memory to the disk */
		vm->storage[original_page_num * 32 + i] = vm->buffer[frame_num * 32 + i];

		/* load the needed page from disk to memory */
		vm->buffer[frame_num * 32 + i] = vm->storage[page_num * 32 + i];

	}

	/* change the page num */
	vm->invert_page_table[frame_num] = page_num;
}

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
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  
	u32 offset = addr % 32;
	u32 page_num = addr / 32;
	u32 frame_num;

	/* Page fault occur, set the first index to the least recent use and replace it */
	if (ISPageFault(vm,page_num)) {
		frame_num = vm->invert_page_table[vm->PAGE_ENTRIES];

		/* load the needed value to memory */
		SwapToMemory(vm, frame_num, page_num);
	}
	else {
		frame_num = FindFrameNum(vm, page_num);
	}

	/* find the frame num and return its value in corresponding meory */
	u32 newframe_num = FindFrameNum(vm, page_num);
	uchar ReturnValue = vm->buffer[newframe_num * 32 + offset];

	/* Update the table */
	UpdateTable(vm, frame_num);

  	return ReturnValue;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {

	u32 offset = addr % 32;
	u32 page_num = addr / 32;
	u32 frame_num;

	if (ISPageFault(vm, page_num)) { 
		/* Page fault occur, set the first index to the least recent index in table */
		frame_num = vm->invert_page_table[vm->PAGE_ENTRIES];

		/* check whether the table is full, if full, swap the page to the disk */
		ISFrameFull(vm, page_num, frame_num);

		/* set the page num to this frame */
		vm->invert_page_table[frame_num] = page_num;
	}
	else {
		/* no page fault occcur, find the index frame num to access the data */
		frame_num = FindFrameNum(vm, page_num);
	}

	vm->buffer[frame_num * 32 + offset] = value;

	/* Update the invert_page_table to maintain its order */
	UpdateTable(vm, frame_num);
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
	int input_size) {

	for (int i = 0; i < input_size; i++) {

		u32 page_num = i / 32;
		u32 offset = i % 32;

		/* read the value from memory and write into results */
      	uchar value = vm_read(vm,i);
      	results[page_num * 32 + offset] = value;
	} 
}

