#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){

	/* fork a child process */
	pid_t pid;
	int status;

	printf("process start to fork \n");
	pid = fork();

	if (pid == -1){
		perror("fork");
		exit(1);
	}

	else{

		if (pid == 0){

			/// Child Process ///

			int i;
			char * arg[argc];
			
			for ( i = 0; i < argc - 1; i++){
				arg[i] = argv[i+1];
			}
			arg[argc - 1] = NULL;

			/* execute test program */
			printf("I'm the Child Process, my pid = %d\n", getpid());
			printf("Children process start to execute the program:\n");
			execve(arg[0], arg,NULL);

			/* check if the child process is replaced by new process */
			printf("Continue to run original child process!\n");
			perror("execve");
			exit(EXIT_FAILURE);

		}

		else{
			 /// Parent Process ///
			
			printf("I'm Parent Process, my pid = %d\n",getpid());

			/* wait for child process terminates */

			waitpid(-1, &status, WUNTRACED);

			printf("Parent process receives SIGCHLD signal\n");

			/* check child process'  termination status */

			/* case 1: normal exit */
			if (WIFEXITED(status)){
				printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
			}

			/* case 2: abnormal exit and send the siganl */
			else if (WIFSIGNALED(status)){
				
				int signal = WTERMSIG(status);

				// abort
				if (signal == 6){
					printf("child process get SIGABRT signal\n");
					printf("child process is abort by abort signal\n");
				}

				// alarm
				else if (signal == 14){
					printf("child process get SIGALRM signal\n");
					printf("child process is abort by alarm signal\n");
					printf("CHILD EXECUTION FAILED");
				}

				// bus
				else if (signal == 7){
					printf("child process get SIGBUS signal\n");
					printf("child process is abort by BUS signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// floating
				else if (signal == 8){
					printf("child process get SIGFPE signal\n");
					printf("child process is abort by SIGFPE signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// hangup
				else if (signal == 1){
					printf("child process get SIGHUP signal\n");
					printf("child process is hung up\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// illgal_instr
				else if (signal == 4){
					printf("child process get SIGILL signal\n");
					printf("child process is abort by SIGILL signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// interrupt
				else if (signal == 2){
					printf("child process get SIGINT signal\n");
					printf("child process is abort by SIGINT signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// kill
				else if (signal == 9){
					printf("child process get SIGKILL signal\n");
					printf("child process is abort by SIGKILL signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// pipe
				else if (signal == 13){
					printf("child process get SIGPIPE signal\n");
					printf("child process is abort by SIGPIPE signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// quit
				else if (signal == 3){
					printf("child process get SIGQUIT signal\n");
					printf("child process is abort by SIGQUIT signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// segment_fault
				else if (signal == 11){
					printf("child process get SIGSEGV signal\n");
					printf("child process is abort by SIGSEGV signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// terminates
				else if (signal == 15){
					printf("child process get SIGTERM signal\n");
					printf("child process is abort by SIGTERM signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}

				// trap
				else if (signal == 5){
					printf("child process get SIGTRAP signal\n");
					printf("child process is abort by SIGTRAP signal\n");
					printf("CHILD EXECUTION FAILED\n");
				}
			}

			/* case 3: stop signal*/
			else if (WIFSTOPPED(status)){
				printf("child process get SIGSTOP signal\n");
				printf("child process stopped\n");
				printf("CHILD EXECUTION STOPPED\n");        
			}

			else{
				printf("CHILD PROCESS CONTINUED\n");
			}

			exit(0);
		}
	}
	
}
