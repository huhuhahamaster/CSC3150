# 3150-Final-Review

**Objective: This is the Final exam review of Operating System course. It mainly focus on the content of OS architecture, structure and  its shecduilng and management.  The Review will cover all the lecture have been taught.**

[TOC]

## Chapter 1: Introduction

### OS Basic Concept

**Computer system can be divided into four component:** 

Hardware(CPU, memory, IO device) --> Operaitng system --> Application Program --> Users

**What is OS and What does OS do:**

- OS is a program acts as an intermediary between a user and computer hardware.
- OS can execute user program and make solving user problems easily (don't care about resource utilization)
- OS can make teh computer system convenient to use
- use the computer hardware in a efficient way. 

**OS is a resource allocator and a control program.**

 Based on that, you should know: The one program runnig at all the time on the computer is the kernel, everything else is either a system program ot an application program. 

**computer startup: bootstrap.** is reloaded at power-up or reboot. Typically stored in ROM（只读内存） or EPROM, generally known as firmware

supplement：what does computer do when you boot it. 

computer system operations: one or more CPUs, device controller connect through common bus providing access to shared memory. **concurrent execution of CPUs and IO devices competing the memory cycles.** 

I/O device can execute concurrently, and  each device controller is in charge of particular device type, have a local buffer. I/O is from the device to local buffer of controller. **Device controller informs CPU that it is finished the operations by causing a interrupt.**

<img src="/Users/huangpengxiang/Library/Application Support/typora-user-images/截屏2021-12-10 上午11.39.08.png" alt="截屏2021-12-10 上午11.39.08" style="zoom:50%;" />

### Interrupts

interrupt transer control to the interrupt service routine generally through a interrupt vetor, which contains the address of all the service routine. 

interrupt architecture must save the address of the interrupted instruction. 

A trap or exception is a soft-ware-generated interrupt casued either by an error or a user request.

An operating system is interrupt driven.

supplement:

中断的作用： CPU上可以运行两种程序， 一种是操作系统内核程序，一种是应用程序。 中断则是会使CPU从用户态转为内核态，使操作系统重新夺回对系统的控制权。

内核态 --> 用户态： 操作系统执行一条特权指令，修改PSW的标志为“用户态”，意味着操作系统主动让出CPU控制权。

用户态 -->内核态： 由“中断”引发， 硬件自动完成转化的过程，触发中断信号意味着操作系统重新夺回控制权。

中断的类型：内中断与外中断。

内中断指的是与执行当前指令有关，中断信号来自操作系统的内部，including trap（由陷入指令引起，应用程序故意引发）, fault（错误条件引起，可能被内核程序修复，修复故障后会把CPU使用权交还给应用程序，例如缺页故障）, abort（由致命错误引起，内核程序无法修复，一般直接终止该程序，如整数除0）外中断与执行当前指令无关，中断信号来自CPU的外部。例如时钟中断和I/O请求中断。

**Interrupt handling:**

The operating system presever the state of the CPU by storing the registers and the program counter. 

determine which type of interrupt has occured. 

- polling
- vertored interrupt system

**IO structure:**

- Synchronous IO: after IO starts, control returens user program only upon IO completion. It will wait instruction idles the CPU until the next interrupt. it will wait loop, at most one IO loop request is outstanding at a time, no simultaneous IO processing.
- Asynchronous IO: after IO start, control will return to user program without waiting for IO completion. Then we have system call -request to the OS to allow user to wait for the IO completion. Device-status table contains entry for each IO device indicating its type, address and state.

**Storage structure:**

Main memory: only large storage media that cpu can access directly, random access, typically volatile. 

secondary storage: extension of main memory that provides you large **nonvolatile** storage capacity. 

magnetic disk: rigid metal or glass platters covered with magnetic recording material. Disk surface is logically divided into tracks, which aere subdivided into sectors. The disk controller determinses the logical interaction between device and computer. 

solid-state disks: faster than magnetic disks, nonvolatile. various technologies, becoming more popular.

**Storage Hierarchy:**

storage systems organized in hierarchy: Speed, Cost, Volatitilty. 

Caching: copying information into faster storage system, main memory can be viewed as a cache for secondary storage.

## Chapter 2 System Structure

what is the system call: programming interface to services provided by the os.

Typically, System call interface maintains a table indexed according to these numbers. The ystem call interface invokes intended system call in OS kernel and returns status of the system call and any return values

```markdown
# example of system call
写一个简单程序，从一个文件读取数据并复制到另一个文件。程序首先需要输入两个文件名称：输入文件名称和输出文件名称

先在屏幕上输出提示信息，再从键盘上读取定义两个文件名称的字符。对于基于鼠标和图标的系统，一个文件名称的菜单通常显示在窗口内。用户通过鼠标选择源文件名称，另一个类似窗口可以用来选择目的文件名称。这个过程需要许多 I/O 系统调用。

在得到两个文件名称后，该程序打开输入文件并创建输出文件。每个操作都需要一个系统调用。每个操作都有可能遇到错误情况，进而可能需要其他系统调用。

例如，当程序设法打开输入文件时，它可能发现该文件不存在或者该文件受保护而不能访问。在这些情况下，程序应在控制台上打印出消息（另一系列系统调用），并且非正常地终止（另一个系统调用）。如果输入文件存在，那么必须创建输出文件。可能发现具有同一名称的输出文件已存在。这种情况可以导致程序中止（一个系统调用），或者可以删除现有文件（另一个系统调用）并创建新的文件（另一个系统调用）。对于交互系统，另一选择是询问用户（一系列的系统调用以输出提示信息并从控制台读人响应）是否需要替代现有文件或中止程序。
```



## Chapter 3 Process Concept

Program is passive entity stored on disk (executable file), process is active. It means that program becomes process when executable file loaded into memory. So **one program can be several processes.**

Execution of program started via GUI mouse clicks, command line entry of its name, etc. 

**Process State:** as a program executes, it changes state. 

new (created), running(executed), waiting, ready(waiting to be assigned to a processor) , terminated.

<img src="/Users/huangpengxiang/Library/Application Support/typora-user-images/截屏2021-12-10 下午7.21.57.png" alt="截屏2021-12-10 下午7.21.57" style="zoom:50%;" />

**Process control block:**

首先进程就是一个运行当中的程序. 程序本来是存储在磁盘的,当我们 需要执行它的时候，先把他读取到内存当中，再然后放入到寄存器中，最后让cpu执行程序，这个时候程序就变成了一个进程. 但是进程的生命周期其 实不是很长，因为程序运行结束之后，进程的生命周期就终止了. 那么每一个进程肯定都是一个独立的个体，那么每个进程与进程直接肯定都拥有自 己独有的一份管理自己的单独的任务结果 .而这个任务结果就是我们今天的PCB. 

<img src="/Users/huangpengxiang/Library/Application Support/typora-user-images/截屏2021-12-10 下午8.29.40.png" alt="截屏2021-12-10 下午8.29.40" style="zoom:50%;" />

**Process Scheduling:**

Process scheduler selects among available processes for next
execution on CPU, and maintain the scheduling queues of processes. 
**context switch:** 

when CPU switch to another process, the system must save the state of the old process and load the saved state for the new process via context swicth.

context of a process represented in the PCB. and context switch time is overhead, the system does not useful work while switching. 

**Process Creation:**

generally, process identified and managed via a process identifier(pid)

parent and children share all resource (copy on write)

**Interprocess communication:**

Process within a system may be independent or cooperating. Cooperating process can affect or be affect by other processes, including sharing data. 

Cooperating process need interprocess communication. (IPC) two models (shared memory and message passing )

**Direct Communication:** process must name each other explicity.

**Indirect Commucation:** Messages are directed and receieved from mailbox. (link established only if processes share a common mailbox, a link must be associated with many processes)

**Synchronization:** during the message passing, there are two way, blocking and non-blocking. blocking can be considered as synchronous, while non-blocking can be asynchronous. 

**Socket:** a socket is defined as an endpoint for communication. Communication consist between a pair of sockets.  

## Chapter 4 Mutithread Programming 

Process creation is heavy-weight while thread creation is light-weight. So it can simplify code, increase effficiency. 

<img src="/Users/huangpengxiang/Library/Application Support/typora-user-images/截屏2021-12-11 上午9.25.50.png" alt="截屏2021-12-11 上午9.25.50" style="zoom:50%;" />

Kernel thread: supported by the kernel ( virtually all general purpose operating system)

User thread: management done by user-level thread library.

many-to-one: many user level trheads mapped to single kernel trhead. one blocking causes all to block, may not run in parallle since one may in kernel at a time. 

One-to-one: each user-level thread maps to kernel thread. Number of threads per process sometimes restricted due to overhead. 

Many-to-many: allow many user level threads to be mapped to many kernel thread. 

**Spin-Lock:** will keep trying to reach untill get the lock.

Implicit Threading: creation and management of threads done by complier and run-time library rather than programmers

## Chaper 5 Process Scheduling

Process execution consist of a cycle of CPU execution and I/O wait. 

CPU burse followed by IO burst.

 there are CPU bound and IO bound, hence we need to design a algorithm to schedule the process. 

CPU schedulling decisions may take place when a process: (non-preemptive)

1. switch from running to waiting queue 
2. Switch from running to ready queue
3. Switch from waiting to ready
4. terminate

**All other is preemptive**

consider access to shared data

consider preemption while in kernel mode

consider interrupts occuring during crucial OS activities.

**Dispatcher:**

gives control of the CPU to the process selected by the short-term scheduler, this involves: 

Switch context

Switch to user mode 

jumping to the proper location in the user program to restart that program.

dispatch latency: time for the dispatcher to stop one process and start another running. 

```markdown
Criteria

# CPU utilization
keep CPU as busy as possible 

# Throughput 
number of process that complete their execution per time unit

# Turnaround time
amount of time to execute a particular process （提交到完成的时间）

# waiting time
amount of time a process has been waiting in the ready queue

# Response time 
amount of time it takes from when request was submitted until the first response is produced, not output (从提交到第一次响应)
```

**Scheduling Algorithm:**

- FCFS: First-come, First-Served.  convey effect: short process behind long processes. (consider one cpu-bound and many IO bound process)

- SJF: shortes job first. associated with each process length of its next cpu burst. Use these lengths to schedule the process with shortest time. 

  **SJF is optimal** give the minimum average waiting time for given set of processes. but the difficulty is konwing the length of the next cpu request. 

- Priority Sheduling: a priority number(interger) is assosiated with each process. 

  Problem--> starvation

  Solution--> Aging

- Round Robin (RR): each process get a small unit of CPU time (time quatum q), If there are n processes get 1/n of the cpu time in chunks of at most q time unit at once.  Timer interrupts every quantum to schedule next process. performance: q large --> FIFO, q small, the overhead is too high. 

**Mutilevel Queue:** 

Ready queue is partitioned into separate queues, eg: Foreground, background.  Process. Each queue has its own scheduling algorithm. Foreground -- RR, Background -- FCFS

Soft real-time system:  no guarantee as to when critital real, time process will be scheduled 

hard real-time system: task must be serviced by its deadline.

two types of latencies affect performance:

**Interrupt latency:** time from arrival of interrupt to start of routine that service interrupt.

**Dispatch latency:** time for schedule switch take current process off cpu and switch to another.

## Chapter 6 Synchronization 

processes can execute concurrently, maybe ineterruped at any time, partially completing execution

Synchronization hardware: many systems provide hardware support for critital section code. Allow solutions below based on idea of locking. 

Modern mahcine provide special atomic hardware instrucitons.

Deadlock: two or more processes are waiting indefinitely for an event that can be casued by only one of the waiting processes.

Starvation: a process may never be removed from the semaphore queue in which it is suspended. 

priority inversion: scheduling problem when lower-priority process holds a lock needed by higher-priority process. 

## Chapter 8 Memory Management

Program must be brought (from disk) into memory and placed within process for it to  be run.  Main memory and register are only storage cpu can access directly. 

A pair of base and limit registers define the logical address space, CPU must check every memory access generated in user mode to be sure it is between base and limit for that user. 

**Address binding:** addresses represented in different way at different stages of a program's life. 

source code address usually symbolic, and compiled code addresses bind to relocatable address. 

can happen in three stages: complied time, load time, execution time. 

The user program deals with logical address, it never sees the real physical address. 

**Dynamic Linking:** Dynamic linking postponed combining program code an system library into binary code until execution time. 

**Swapping:** A process can be swapped temporarily out of the memory to backing store, and then brought back into memory for continued execution. 

context switch time can then be very high. 

constrains on swapping: pending IO, can't swap as IO would occur to wrong process, or always transfer IO to kernel space, then to IO device. 

**Main Memory usually into two partitions:** Resident operating system, usually hed in low memory with interrupt vector, user process then held in high memory, each process contain in single contiguous section of memory.

**Dynamic Storage-allocation Problem**: How to satisfy a request of size n from a list of free holes. First fit, Best fit, Worst fit. First fit and best-fit better than worst fit in terms of spee and storage utilization. 

**Reduce external fragmentation by compaction.** Shuffle memory contents to place all free memory together in one large block.  it is only possible only if relocaton is dynamic, and is done at executiontime. IO probelm also exist, some process will latch the job in memory while it involved IO. Do IO only into OS buffer.

**Reduce internal fragmentation by segmentation.** Logical address contain a two tuple, <segment-number, offset>. Use two register to represent the segment table's infomation(base, limit). 

**Paging:** way to reduce external fragmentation. 

divide physical memory into fixed-size blocks called frames, divide logical memory into blocks of same size called pages. 

**Memory Protection:** valid-invalid bit to indecated the legal or illegal instruction, can also implemented by page-lenghth register. 

## Chapter 9 Virtual-Memory Management

**Demanding page:** could bring entire process into memory at lod time, or bring a page into memory only when it is needed. 

**Page fault:** if there is a reference to a page, first reference to that page will trap to operating system: page fault. 

**Page replacement:** prevent over-allocation of memory by modifying page-fault service routine to include page replacement. 

Find a free frame: 

If there is a free frame, use it

if there is no free frame, use a page replacement algorithm to select a victim frame

Write victim frame to disk if dirty.

FIFO algorithm: first in first out algorithm. 

OPT algorithm: replece page taht will not be used for longest period of time.

LRU algorithm: least recent used algorithm. Every page has a counter, every time page is referenced through this entry, copy the clock into counter. 

**Thrashing:** If a process does not have enough pages, the page fault rate is very high. (page fault to get page, replace existing frame, be quickly need replacced frame back). This will lead to low cpu utilization, operating system thinking that it needs to increase the degree of mutiprogramming, another process added to the system. Thrashing means a process is busying swapping pages in and out. 

Global replacement: process select a replacement frame from the set of all frames, one process can take a frame from another. 

Local replacement: each processes selects from only its own set of allocated frames. more consistent per process performance. 

**Memory-Mapped Files:** memory-mapped file IO allows file IO to be treated as routine memory access by mapping a disk block a page in memory. 

## Chapter 10 File System

Contents defined by file creator. There are many types like text file, source file, executable file. 

File Operations: file is an abstract type,  create, wirte, read, delete, etc.

Disk can be subdivided into partitions, disk or partition can be RAID protected against failure. 

文件系统应该放在非易失系统当中

文件是信息的逻辑存储单位

File Attributes: name, identifier, type, location, and so on. 

Sequential access of file: mantain the pointer, read or write in order, can rewind. 

Index and relative files: involve creation of an index for the file. 

**Disk Structure:** disk can be divided into partition. Entity containing file system knows as a volume, each volume containing file system also track that file system infor in device directory or volume table of contents.

Directory structure: single-level, two-level, tree-structured. 

**File sharing:** sharing file on multi-user system through a protection scheme. NFS (network file sytem)

## Chapter 11 Implementation File System

File system structure: Logical storage unit.

partition and moutings: partiton can. be a volume containing a file system or raw, jsut a sequence of blocks with no file system. boot block can point to boot volume or boot loader set of blocks that contain enough code to know how to load the kernel from the file system. 

VFS: allows the same system call interface (the API) to be used for different types of file system. 

Directory implementation: Linear list of file names with pointer to the data blocks; Hash Table. 

**Allocation method:** an allocation method refers to how disk blocks are allocated for files.  

Conriguous allocation. each file occupies set of contiguous blocks, best performance in most cases. simple, only need start location and lenghth. problem include finding sapce for file, knowing file size, need for compaction.

Linked allocation: maintain the pointer, but esay to lose the file, so use FAT method to maintain the relationship between each block.

Indexed: each file has its own index blocks of pointers to its data blocks.allocate more block to store the index. need index table, random access, dynamic access without external fragmentation, but have overhead of index block.

## Chapter 13 IO

Concept: Port, Bus, Conroller(host adapter) 

Polling: Read bust bit from status register untill 0. but it is inefficient since device is slow.

Interrupts: CPU interrupt-request line --> interrupt handler --> interrupte vetor to dispatch interrupt to correct handler. 

DMA: direct memory access.



