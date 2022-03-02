#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

void fork_tree(int argc, char *argv[],int iterator);
void print_process(int num);
void print_status(int nums);
void signal_mapping(int child_process, int parent_process, int signal);

static int *process_tree;
static int *status_tree;
static int *status_count;

int main(int argc,char *argv[]){


	int filenum = argc - 1;

	/* if only one argument file in command line, print the pid and return */
	if (filenum == 0){
		printf("the process tree: %d\n", getpid());
		printf("MyFork process(pid=%d) execute normally\n", getpid());
		return 0;
	}

	/* using Memory Mapped files for Process communication  */
	process_tree = mmap(NULL, sizeof(int)*(argc), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    status_tree = mmap(NULL, sizeof(int)*(argc), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    status_count = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	*status_count = argc -1;
	*(process_tree) = getpid();

	/* store the number of iterations */
	int index = 0;

	int i;
	char * arg[argc];
	for(i=0; i<argc-1; i++){
        arg[i] = argv[i+1];
    }
	arg[argc-1] = NULL;

	/* generate a fork tree*/
	fork_tree(argc, arg,index);

	print_process(filenum);
	print_status(filenum);
	
	printf("Myfork process (%d) terminated normally \n", *(process_tree));

	return 0;
}

void fork_tree(int argc, char *argv[], int index){
	int status;
	pid_t pid;
	
	pid = fork();

	if (pid < 0){
		printf ("Fork error!\n");
	}

	else{

		// child process
		if (pid == 0) {
			if (index  == argc - 1){
				exit(0);
			}
			else{
				int mypid = getpid();
				*(process_tree + index + 1) = mypid;

				index++;
				fork_tree(argc,argv,index);
				execve(argv[index-1],NULL,NULL);
			}
		}

		// parent process
		else{
			waitpid(-1, &status, WUNTRACED);
			if (status == 0) {
				*(status_tree + index) = status;
			}
			else{
				*(status_tree + index) = WTERMSIG(status);
			}
		}
	}
}

void print_process(int nums){
	int i = 0;
	printf("the process tree: ");
	for ( i = 0; i < nums; i++){
		printf("%d->", *(process_tree+i));
	}
	printf("%d\n", *(process_tree+i));
}

void print_status(int nums){
	for (int j = nums - 1; j > -1; j--){
		signal_mapping(*(process_tree+j+1),*(process_tree+j),*(status_tree+j));
	}
}

void signal_mapping(int child_process, int parent_process, int signal){
	if (signal == 0){
		printf("Child process %d of parent process %d terminated normally with exit code %d\n",child_process,parent_process, signal);
	}

	// abort
	else if (signal == 6){
		printf("Child process %d of parent process %d is terminated by signal %d (Abort) \n",child_process,parent_process, signal);
	}

	// alarm
	else if (signal == 14){
		printf("Child process %d of parent process %d is terminated by signal %d (Alarm clock) \n",child_process,parent_process, signal);
	}

	//bus
	else if (signal == 7){
		printf("Child process %d of parent process %d is terminated by signal %d (Bus) \n",child_process,parent_process, signal);
	}

	// floating
	else if (signal == 8){
		printf("Child process %d of parent process %d is terminated by signal %d (Floating) \n",child_process,parent_process, signal);
	}

	// hangup
	else if (signal == 1){
		printf("Child process %d of parent process %d is terminated by signal %d (Hangup) \n",child_process,parent_process, signal);
	}

	// illgal_instr
	else if (signal == 4){
		printf("Child process %d of parent process %d is terminated by signal %d (Illgal_instr) \n",child_process,parent_process, signal);
	}

	// interrupt
	else if (signal == 2){
		printf("Child process %d of parent process %d is terminated by signal %d (Interrupt) \n",child_process,parent_process, signal);
	}

	// kill
	else if (signal == 9){
		printf("Child process %d of parent process %d is terminated by signal %d (Kill) \n",child_process,parent_process, signal);
	}

	// pipe
	else if (signal == 13){
		printf("Child process %d of parent process %d is terminated by signal %d (PIPE) \n",child_process,parent_process, signal);
	}

	// quit
	else if (signal == 3){
		printf("Child process %d of parent process %d is terminated by signal %d (Quit) \n",child_process,parent_process, signal);
	}

	else if (signal == 11){
		printf("Child process %d of parent process %d is terminated by signal %d (Segment_fault) \n",child_process,parent_process, signal);
	}

	// terminate 
	else if (signal == 15){
		printf("Child process %d of parent process %d is terminated by signal %d (Terminate) \n",child_process,parent_process, signal);
	}

	// trap
	else if (signal == 5){
		printf("Child process %d of parent process %d is terminated by signal %d (Trap) \n",child_process,parent_process, signal);
	}

	else{
		printf("no such signal to handle in this program. Terminated\n");
		printf("unknown terminated signal is: %d \n", signal);
		exit(1);
	}
}
