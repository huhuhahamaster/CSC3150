#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>		
#include <linux/fs.h>
#include <linux/cdev.h>	
#include <asm/uaccess.h>	
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/slab.h>

static int dev_major;
static int dev_minor;

static int __init init_modules(void)
{
	dev_t dev;
	int ret = 0;

	printk("Tutorial_10:%s():..............Start..............\n",__FUNCTION__);
	ret = alloc_chrdev_region(&dev, 0, 1, "mydev");
	if(ret)
	{
		printk("Cannot alloc chrdev\n");
		return ret;
	}
	
	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);
	printk("Tutorial_10:%s():register chrdev(%d,%d)\n",__FUNCTION__,dev_major,dev_minor);
	

	return 0;	
}

static void __exit exit_modules(void)
{

	printk("Tutorial_10:%s():..............End..............\n",__FUNCTION__);
}

module_init(init_modules);
module_exit(exit_modules);
