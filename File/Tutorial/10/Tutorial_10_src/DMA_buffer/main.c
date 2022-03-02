#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>		
#include <linux/fs.h>
#include <linux/cdev.h>	
#include <asm/uaccess.h>	
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/slab.h>

void *dma_buf;
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdevp;


static int drv_open(struct inode *inode, struct file *filp);
static int drv_release(struct inode *inode, struct file *filp);
static ssize_t drv_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos);
static ssize_t drv_write(struct file *filp, const char __user *buf, size_t count, loff_t *f_pos);
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);

#define DMA_BUFSIZE 64

struct file_operations drv_fops = 
{
	owner:	THIS_MODULE,
	read:	drv_read,
	write:	drv_write,
	unlocked_ioctl:	drv_ioctl,
	open:	drv_open,
	release:	drv_release
};

static int drv_open(struct inode *inode, struct file *filp)
{
	printk("Tutorial_10:%s():device open\n", __FUNCTION__);
	return 0;
}

static int drv_release(struct inode *inode, struct file *filp)
{
	printk("Tutorial_10:%s():device close\n",__FUNCTION__);
	return 0;
}

static ssize_t drv_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos)
{
	printk("Tutorial_10:%s(): in read\n", __FUNCTION__);
	return 0;
}



static ssize_t drv_write(struct file *filp, const char __user *buf, size_t count, loff_t *f_pos)
{
	printk("Tutorial_10:%s(): in write\n", __FUNCTION__);
	return 0;
}

static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	printk("Tutorial_10:%s(): in ioctl\n", __FUNCTION__);
	return 0;

}



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
	printk("Tutorial_9:%s():register chrdev(%d,%d)\n",__FUNCTION__,dev_major,dev_minor);
	

	dev_cdevp = cdev_alloc();

	cdev_init(dev_cdevp, &drv_fops);
	dev_cdevp->owner = THIS_MODULE;
	ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), 1);
	if(ret < 0)
	{
		printk("Add chrdev failed\n");
		return ret;
	}

	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("Tutorial_10:%s():allocate dma buffer\n",__FUNCTION__);

	return 0;	
}

static void __exit exit_modules(void)
{
	dev_t dev;
	
	dev = MKDEV(dev_major, dev_minor);
	cdev_del(dev_cdevp);

	kfree(dma_buf);
	printk("Tutorial_10:%s():free dma buffer\n",__FUNCTION__);

	printk("Tutorial_10:%s():unregister chrdev\n",__FUNCTION__);
	unregister_chrdev_region(dev, 1);

	printk("Tutorial_10:%s():..............End..............\n",__FUNCTION__);
}

module_init(init_modules);
module_exit(exit_modules);
