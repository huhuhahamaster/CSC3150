#include <linux/init.h>
#include <linux/module.h>
#include <linux/kthread.h>

MODULE_LICENSE("GPL");


static struct task_struct *task;



//implement test function
int func(void* data)  {
  
    int time_count = 0;
	do {
		printk(KERN_INFO "thread_function: %d times", ++time_count);


	}while(!kthread_should_stop() && time_count<=30);

	return time_count; 
}  


static int __init KT_init(void){
	
	printk("KT module create kthread start\n");

	//create a kthread
	task=kthread_create(&func,NULL,"MyThread");


	//wake up new thread if ok
	if(!IS_ERR(task)){
		printk("Kthread starts\n");
		wake_up_process(task);
	}
	return 0;
}

static void __exit KT_exit(void){
	printk("KT module exits! \n");
}

module_init(KT_init);
module_exit(KT_exit);
