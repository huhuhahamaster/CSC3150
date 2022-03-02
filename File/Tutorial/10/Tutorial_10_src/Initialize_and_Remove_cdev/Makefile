mname := mydev
$(mname)-objs := main.o
obj-m := $(mname).o

KVERSION := $(shell uname -r)
PWD	:= $(shell pwd)

all:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) modules
clean:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) clean
