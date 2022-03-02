#include <linux/init.h>
#include <linux/module.h>

MODULE_LICENSE("GPL");

static int KM_init(void) {
    printk(KERN_INFO "Kernel Module initilization!\n");

    return 0;
}

static void KM_exit(void) {
    printk(KERN_INFO "Kernel Module exits!\n");

}

module_init(KM_init);
module_exit(KM_exit);
