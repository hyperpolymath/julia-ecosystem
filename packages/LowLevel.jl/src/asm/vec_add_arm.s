// AArch64 implementation of vector addition
.global vec_add_arm
.text

vec_add_arm:
    // x0 = ptr A, x1 = ptr B, x2 = ptr Result, x3 = length
    cbz x3, .done
.loop:
    ldr x4, [x0], #8
    ldr x5, [x1], #8
    add x6, x4, x5
    str x6, [x2], #8
    subs x3, x3, #1
    b.ne .loop
.done:
    ret
