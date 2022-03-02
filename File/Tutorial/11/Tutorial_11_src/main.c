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

MODULE_LICENSE("GPL");

// CONSTS
#define PREFIX_TITLE "Tutorial_11"

// DEVICE
#define DEV_NAME "mydev"        // name for alloc_chrdev_region
#define DEV_BASEMINOR 0         // baseminor for alloc_chrdev_region
#define DEV_COUNT 1             // count for alloc_chrdev_region
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;


// File Operations
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static struct file_operations fops = {
      	owner: THIS_MODULE,
	write: drv_write,
	open: drv_open,
      	release: drv_release,
};

// Work routine
static struct work_struct *work;

// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}

static int drv_release(struct inode* ii, struct file* ff) 
{
	module_put(THIS_MODULE);
	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}

static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {

	int IOMode;
 	get_user(IOMode, (int *) buffer);
	printk("%s:%s(): IO Mode is %d\n", PREFIX_TITLE, __func__, IOMode);

	INIT_WORK(work, drv_arithmetic_routine);

	// Decide io mode
	if(IOMode) {
		// Blocking IO
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		schedule_work(work);
		flush_scheduled_work();
    	} 
	else {
		// Non-locking IO
		printk("%s,%s(): non-blocking\n",PREFIX_TITLE, __func__);
		schedule_work(work);
   	 }
	return 0;
}


static void drv_arithmetic_routine(struct work_struct* ws) 
{

	printk("In routine!\n");
	msleep(5000);
	printk("Work routine completed!\n");
}

static int __init init_modules(void) 
{

	dev_t dev;

	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);
  	
	dev_cdev = cdev_alloc();

	// Register chrdev
	if(alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME) < 0) {
		printk(KERN_ALERT"Register chrdev failed!\n");
		return -1;
    	} else {
		printk("%s:%s(): register chrdev(%i,%i)\n", PREFIX_TITLE, __func__, MAJOR(dev), MINOR(dev));
    	}

    	dev_major = MAJOR(dev);
    	dev_minor = MINOR(dev);

	// Init cdev
   	dev_cdev->ops = &fops;
    	dev_cdev->owner = THIS_MODULE;

	if(cdev_add(dev_cdev, dev, 1) < 0) {
		printk(KERN_ALERT"Add cdev failed!\n");
		return -1;
   	}


	// Alloc work routine
    	work = kmalloc(sizeof(typeof(*work)), GFP_KERNEL);

	return 0;
}

static void __exit exit_modules(void) {

	// Delete char device
	unregister_chrdev_region(MKDEV(dev_major,dev_minor), DEV_COUNT);
	cdev_del(dev_cdev);

	// Free work routine
	kfree(work);
	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);
	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}


module_init(init_modules);
module_exit(exit_modules);
