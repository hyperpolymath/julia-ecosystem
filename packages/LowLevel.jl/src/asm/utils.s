; src/asm/utils.s
global asm_add

section .text
asm_add:
    ; Arguments are in rdi (a), rsi (b)
    mov rax, rdi
    add rax, rsi
    ret
