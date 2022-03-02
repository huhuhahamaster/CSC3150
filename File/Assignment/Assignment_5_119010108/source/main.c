#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"

// define the IRQ number 
#define IRQ_NUM 1

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2

void *dma_buf;
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;
static int interruptcount = 0; // count the keyboard interrupt 
static int keyBoardISR_devID = 1;

static void handler(void);

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;


// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);


// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}
static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement read operation for your device */
	int result;
	result = myini(DMAANSADDR);	
	put_user(result, (int*) buffer);
	printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, result);

	// reset the readable to false 
	myouti(0, DMAREADABLEADDR);

	// clean the value 
	myouti(0, DMASTUIDADDR);
	myouti(0, DMARWOKADDR);
	myouti(0, DMAIOCOKADDR);
	myouti(0, DMAIRQOKADDR);
	myouti(0, DMACOUNTADDR);
	myouti(0, DMAANSADDR);
	myouti(0, DMABLOCKADDR);
	myouti(0, DMAOPCODEADDR);
	myoutc(NULL, DMAOPCODEADDR);
	myouti(0, DMAOPERANDCADDR);
	myouti(0, DMAOPERANDBADDR);

	return 0;
}
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement write operation for your device */
	struct DataIn datain;

	get_user(datain.a, (char*)buffer );
	get_user(datain.b, (int*)buffer + 1);
	get_user(datain.c, (int*)buffer + 2);

	myoutc(datain.a, DMAOPCODEADDR);
	myouti(datain.b, DMAOPERANDBADDR);
	myouti(datain.c, DMAOPERANDCADDR);

	int mode = myini(DMABLOCKADDR);

	/* initial the calculate */
	INIT_WORK(work_routine,drv_arithmetic_routine);

	printk("%s:%s():queue work\n",PREFIX_TITLE,__func__);

	/* check the mode and schedule the work */
	if (mode == 1){
		printk("%s:%s():block\n",PREFIX_TITLE,__func__);
		schedule_work(work_routine);
		flush_scheduled_work();
	}else{
		printk("%s,%s(): non-blocking\n",PREFIX_TITLE,__func__);
		myouti(0, DMAREADABLEADDR);
		schedule_work(work_routine);
	}
	return 0;
}
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	
	/* get the user input and readable info */
	int info;
	get_user(info,(int*)arg);
	int readable  = myini(DMAREADABLEADDR);

	switch (cmd) {
		case HW5_IOCSETSTUID:
			myouti(info, DMASTUIDADDR);
			printk("%s:%s(): My STUID is: %d\n", PREFIX_TITLE, __func__, info);
			break;
		case HW5_IOCSETRWOK:
			myouti(info, DMARWOKADDR);
			if(info == 1 ) printk("%s:%s(): RW OK\n", PREFIX_TITLE, __func__);
			break;
		case HW5_IOCSETIOCOK:
			myouti(info, DMAIOCOKADDR);
			if (info == 1) printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
			break;
		case HW5_IOCSETIRQOK:
			myouti(info, DMAIRQOKADDR);
			if (info == 1) printk("%s:%s(): IRQ OK\n", PREFIX_TITLE, __func__);
			break;
		case HW5_IOCSETBLOCK:
			myouti(info, DMABLOCKADDR);
			if (info == 1) printk("%s:%s(): Blocking IO \n", PREFIX_TITLE, __func__);
			else if (info == 0) printk("%s:%s(): Non-Blocking IO \n", PREFIX_TITLE, __func__);
			break;
		case HW5_IOCWAITREADABLE:
			printk("%s:%s(): wait readable %d \n", PREFIX_TITLE, __func__, 1);
			while (readable == 0){ // while not readable, wait till readable 
				msleep(1000);		 
				readable = myini(DMAREADABLEADDR);
			}
			put_user(readable,(int *)arg);
			break;
		default:
			printk("no such operations \n");
			return -1;
	}

	return 0;
}

static int prime(int base, short nth){
	int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }
        
        if(isPrime) {
            fnd++;
        }
    }

    return num;
}

static void drv_arithmetic_routine(struct work_struct* ws) {

	/* get the input of operator and operand */
	char op = myinc(DMAOPCODEADDR);
	int oprand1 = myini(DMAOPERANDBADDR);
	short oprand2 = myini(DMAOPERANDCADDR);

	int ans;

	/* calculate the answer */
	switch (op) {
		case '+':
			ans= oprand1 + oprand2;
			break;
		case '-' :
			ans= oprand1 - oprand2;
			break;
		case '*':
			ans= oprand1 * oprand2;
			break;
		case '/':
			ans = oprand1 / oprand2;
			break;
		case 'p':
			ans = prime(oprand1, oprand2);
			break;
		default:
			ans = 0;
	}

	/* output the result */
	myouti(ans,DMAANSADDR);
	myouti(1, DMAREADABLEADDR); // set readable to true

	printk("%s:%s(): %i %c %i = %i\n", PREFIX_TITLE,__func__,oprand1, op, oprand2, ans);
}

static void handler(void){
   interruptcount++; // count the keyboard interrupt
}

static int __init init_modules(void) {
    
	int ret = 0;

	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

	/* add the ISR to keyboard IRQ */
	interruptcount = 0;
	ret = request_irq(IRQ_NUM, (irq_handler_t) handler, IRQF_SHARED, "InterruptCount", keyBoardISR_devID);
	if (ret) {
		printk(KERN_ALERT"IRQ initialize failed\n");
		return ret;
	}


	/* Register chrdev */ 
	dev_t dev;
	ret = alloc_chrdev_region(&dev, 0, 1, "mydev");
	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);

	if(ret){
		printk(KERN_ALERT"Register chrdev failed!\n");
		return ret;
	}else{
		printk("%s:%s():register chrdev(%d, %d)\n", PREFIX_TITLE, __func__, dev_major, dev_minor);
	}

	/* Init cdev and make it alive */
	dev_cdev = cdev_alloc();
	cdev_init(dev_cdev, &fops);
	dev_cdev->ops = &fops;
	dev_cdev->owner = THIS_MODULE;
	ret = cdev_add(dev_cdev,dev,1);

	if (ret < 0) {
		printk(KERN_ALERT"%s:%s():Add cdev failed!\n",PREFIX_TITLE,__func__);
		return ret;
	}

	/* Allocate DMA buffer */
	dma_buf = kmalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("%s:%s(): allocate dma buffer\n",PREFIX_TITLE,__func__);

	/* Allocate work routine */
	work_routine =  kmalloc(sizeof(typeof(*work_routine)),GFP_KERNEL);

	return 0;
}

static void __exit exit_modules(void) {
	/* Free IRQ */
    free_irq(IRQ_NUM, keyBoardISR_devID);
    printk("%s:%s():Interrupt count = %d\n", PREFIX_TITLE, __FUNCTION__, interruptcount);

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);

	/* Delete character device */
	unregister_chrdev_region(MKDEV(dev_major,dev_minor),1);
	cdev_del(dev_cdev);

	/* Free work routine */
	kfree(work_routine);
    	printk("%s:%s():unregister chrdev\n",PREFIX_TITLE,__func__);

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
