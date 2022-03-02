## File System





**what is included in this project**

Filesystem: a pointer

volume: The whole memory which inlcude super block + FCB + File size 

- super block:  4KB totally, each bit represent the block is occupied or not.
- FCB: since the maximum file number is 1024, the max FCB entries is 1024, each occupied 32 byte. store the file information
- file size: The real file located, and each file may have different length. need to maintain it order



**what need to be done in this project**

First is Open operation:

The Open operation is need to find the file information by name if it exist in FCB. If not, then need to create a 

new file and give the user a file pointer which is **the first block location in volume of file**.  If it is exist, Then 

need to return the file pointer. 



Second is Read operation:

The read operation is given the file pointer and the size, then read it from volume to output buffer. 



Third is Write operation:

Give the file pointer, and need to write the information into the special file from input to volume.



**The Difficult Part:**

how to maintain the order between FCB and super block, how to compact the file if there is some confliction.  





