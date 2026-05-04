
test_vector_ops_rv64:	file format elf64-littleriscv

Disassembly of section .text:

0000000080000000 <_start>:
; _start():
80000000: 60000293     	li	t0, 0x600
80000004: 3002a073     	csrs	mstatus, t0
80000008: 00003117     	auipc	sp, 0x3
8000000c: 37810113     	addi	sp, sp, 0x378
80000010: 00001297     	auipc	t0, 0x1
80000014: 37028293     	addi	t0, t0, 0x370
80000018: 00001317     	auipc	t1, 0x1
8000001c: 36830313     	addi	t1, t1, 0x368
80000020: 0062f663     	bgeu	t0, t1, 0x8000002c <_start+0x2c>
80000024: 0002b023     	sd	zero, 0x0(t0)
80000028: 02a1         	addi	t0, t0, 0x8
8000002a: bfdd         	j	0x80000020 <_start+0x20>
8000002c: 0ea000ef     	jal	0x80000116 <main>
80000030: 0506         	slli	a0, a0, 0x1
80000032: 00156513     	ori	a0, a0, 0x1
80000036: 00001297     	auipc	t0, 0x1
8000003a: 30a28293     	addi	t0, t0, 0x30a
8000003e: 00a2b023     	sd	a0, 0x0(t0)
80000042: a001         	j	0x80000042 <_start+0x42>

0000000080000044 <memset>:
; memset():
80000044: 7139         	addi	sp, sp, -0x40
80000046: fc06         	sd	ra, 0x38(sp)
80000048: f822         	sd	s0, 0x30(sp)
8000004a: 0080         	addi	s0, sp, 0x40
8000004c: fea43423     	sd	a0, -0x18(s0)
80000050: feb42223     	sw	a1, -0x1c(s0)
80000054: fcc43c23     	sd	a2, -0x28(s0)
80000058: fe843503     	ld	a0, -0x18(s0)
8000005c: fca43823     	sd	a0, -0x30(s0)
80000060: 4501         	li	a0, 0x0
80000062: fca43423     	sd	a0, -0x38(s0)
80000066: a009         	j	0x80000068 <memset+0x24>
80000068: fc843503     	ld	a0, -0x38(s0)
8000006c: fd843583     	ld	a1, -0x28(s0)
80000070: 02b57363     	bgeu	a0, a1, 0x80000096 <memset+0x52>
80000074: a009         	j	0x80000076 <memset+0x32>
80000076: fe444503     	lbu	a0, -0x1c(s0)
8000007a: fd043583     	ld	a1, -0x30(s0)
8000007e: fc843603     	ld	a2, -0x38(s0)
80000082: 95b2         	add	a1, a1, a2
80000084: 00a58023     	sb	a0, 0x0(a1)
80000088: a009         	j	0x8000008a <memset+0x46>
8000008a: fc843503     	ld	a0, -0x38(s0)
8000008e: 0505         	addi	a0, a0, 0x1
80000090: fca43423     	sd	a0, -0x38(s0)
80000094: bfd1         	j	0x80000068 <memset+0x24>
80000096: fe843503     	ld	a0, -0x18(s0)
8000009a: fc040113     	addi	sp, s0, -0x40
8000009e: 70e2         	ld	ra, 0x38(sp)
800000a0: 7442         	ld	s0, 0x30(sp)
800000a2: 6121         	addi	sp, sp, 0x40
800000a4: 8082         	ret

00000000800000a6 <memcpy>:
; memcpy():
800000a6: 7139         	addi	sp, sp, -0x40
800000a8: fc06         	sd	ra, 0x38(sp)
800000aa: f822         	sd	s0, 0x30(sp)
800000ac: 0080         	addi	s0, sp, 0x40
800000ae: fea43423     	sd	a0, -0x18(s0)
800000b2: feb43023     	sd	a1, -0x20(s0)
800000b6: fcc43c23     	sd	a2, -0x28(s0)
800000ba: fe843503     	ld	a0, -0x18(s0)
800000be: fca43823     	sd	a0, -0x30(s0)
800000c2: fe043503     	ld	a0, -0x20(s0)
800000c6: fca43423     	sd	a0, -0x38(s0)
800000ca: 4501         	li	a0, 0x0
800000cc: fca43023     	sd	a0, -0x40(s0)
800000d0: a009         	j	0x800000d2 <memcpy+0x2c>
800000d2: fc043503     	ld	a0, -0x40(s0)
800000d6: fd843583     	ld	a1, -0x28(s0)
800000da: 02b57663     	bgeu	a0, a1, 0x80000106 <memcpy+0x60>
800000de: a009         	j	0x800000e0 <memcpy+0x3a>
800000e0: fc843503     	ld	a0, -0x38(s0)
800000e4: fc043603     	ld	a2, -0x40(s0)
800000e8: 9532         	add	a0, a0, a2
800000ea: 00054503     	lbu	a0, 0x0(a0)
800000ee: fd043583     	ld	a1, -0x30(s0)
800000f2: 95b2         	add	a1, a1, a2
800000f4: 00a58023     	sb	a0, 0x0(a1)
800000f8: a009         	j	0x800000fa <memcpy+0x54>
800000fa: fc043503     	ld	a0, -0x40(s0)
800000fe: 0505         	addi	a0, a0, 0x1
80000100: fca43023     	sd	a0, -0x40(s0)
80000104: b7f9         	j	0x800000d2 <memcpy+0x2c>
80000106: fe843503     	ld	a0, -0x18(s0)
8000010a: fc040113     	addi	sp, s0, -0x40
8000010e: 70e2         	ld	ra, 0x38(sp)
80000110: 7442         	ld	s0, 0x30(sp)
80000112: 6121         	addi	sp, sp, 0x40
80000114: 8082         	ret

0000000080000116 <main>:
; main():
80000116: 1101         	addi	sp, sp, -0x20
80000118: ec06         	sd	ra, 0x18(sp)
8000011a: e822         	sd	s0, 0x10(sp)
8000011c: 1000         	addi	s0, sp, 0x20
8000011e: 4501         	li	a0, 0x0
80000120: fea42623     	sw	a0, -0x14(s0)
80000124: 0a8000ef     	jal	0x800001cc <test_relu_i8>
80000128: c511         	beqz	a0, 0x80000134 <main+0x1e>
8000012a: a009         	j	0x8000012c <main+0x16>
8000012c: 4505         	li	a0, 0x1
8000012e: fea42623     	sw	a0, -0x14(s0)
80000132: a069         	j	0x800001bc <main+0xa6>
80000134: 160000ef     	jal	0x80000294 <test_relu_i16>
80000138: c511         	beqz	a0, 0x80000144 <main+0x2e>
8000013a: a009         	j	0x8000013c <main+0x26>
8000013c: 4509         	li	a0, 0x2
8000013e: fea42623     	sw	a0, -0x14(s0)
80000142: a8ad         	j	0x800001bc <main+0xa6>
80000144: 202000ef     	jal	0x80000346 <test_elemwise_add_i8>
80000148: c511         	beqz	a0, 0x80000154 <main+0x3e>
8000014a: a009         	j	0x8000014c <main+0x36>
8000014c: 450d         	li	a0, 0x3
8000014e: fea42623     	sw	a0, -0x14(s0)
80000152: a0ad         	j	0x800001bc <main+0xa6>
80000154: 29c000ef     	jal	0x800003f0 <test_elemwise_add_i32>
80000158: c511         	beqz	a0, 0x80000164 <main+0x4e>
8000015a: a009         	j	0x8000015c <main+0x46>
8000015c: 4511         	li	a0, 0x4
8000015e: fea42623     	sw	a0, -0x14(s0)
80000162: a8a9         	j	0x800001bc <main+0xa6>
80000164: 34a000ef     	jal	0x800004ae <test_elemwise_mul_i8>
80000168: c511         	beqz	a0, 0x80000174 <main+0x5e>
8000016a: a009         	j	0x8000016c <main+0x56>
8000016c: 4515         	li	a0, 0x5
8000016e: fea42623     	sw	a0, -0x14(s0)
80000172: a0a9         	j	0x800001bc <main+0xa6>
80000174: 3e4000ef     	jal	0x80000558 <test_elemwise_mul_scalar_i16>
80000178: c511         	beqz	a0, 0x80000184 <main+0x6e>
8000017a: a009         	j	0x8000017c <main+0x66>
8000017c: 4519         	li	a0, 0x6
8000017e: fea42623     	sw	a0, -0x14(s0)
80000182: a82d         	j	0x800001bc <main+0xa6>
80000184: 482000ef     	jal	0x80000606 <test_quantize_i32_to_i8>
80000188: c511         	beqz	a0, 0x80000194 <main+0x7e>
8000018a: a009         	j	0x8000018c <main+0x76>
8000018c: 451d         	li	a0, 0x7
8000018e: fea42623     	sw	a0, -0x14(s0)
80000192: a02d         	j	0x800001bc <main+0xa6>
80000194: 516000ef     	jal	0x800006aa <test_dequantize_i8_to_i32>
80000198: c511         	beqz	a0, 0x800001a4 <main+0x8e>
8000019a: a009         	j	0x8000019c <main+0x86>
8000019c: 4521         	li	a0, 0x8
8000019e: fea42623     	sw	a0, -0x14(s0)
800001a2: a829         	j	0x800001bc <main+0xa6>
800001a4: 5b0000ef     	jal	0x80000754 <test_bias_add>
800001a8: c511         	beqz	a0, 0x800001b4 <main+0x9e>
800001aa: a009         	j	0x800001ac <main+0x96>
800001ac: 4525         	li	a0, 0x9
800001ae: fea42623     	sw	a0, -0x14(s0)
800001b2: a029         	j	0x800001bc <main+0xa6>
800001b4: 4501         	li	a0, 0x0
800001b6: fea42623     	sw	a0, -0x14(s0)
800001ba: a009         	j	0x800001bc <main+0xa6>
800001bc: fec42503     	lw	a0, -0x14(s0)
800001c0: fe040113     	addi	sp, s0, -0x20
800001c4: 60e2         	ld	ra, 0x18(sp)
800001c6: 6442         	ld	s0, 0x10(sp)
800001c8: 6105         	addi	sp, sp, 0x20
800001ca: 8082         	ret

00000000800001cc <test_relu_i8>:
; test_relu_i8():
800001cc: 7139         	addi	sp, sp, -0x40
800001ce: fc06         	sd	ra, 0x38(sp)
800001d0: f822         	sd	s0, 0x30(sp)
800001d2: 0080         	addi	s0, sp, 0x40
800001d4: 08000513     	li	a0, 0x80
800001d8: fea40523     	sb	a0, -0x16(s0)
800001dc: 6521         	lui	a0, 0x8
800001de: f0550513     	addi	a0, a0, -0xfb
800001e2: fea41423     	sh	a0, -0x18(s0)
800001e6: 010105b7     	lui	a1, 0x1010
800001ea: 15ed         	addi	a1, a1, -0x5
800001ec: feb42223     	sw	a1, -0x1c(s0)
800001f0: 4581         	li	a1, 0x0
800001f2: fcb43423     	sd	a1, -0x38(s0)
800001f6: feb40123     	sb	a1, -0x1e(s0)
800001fa: fea41023     	sh	a0, -0x20(s0)
800001fe: 01000537     	lui	a0, 0x1000
80000202: fca42e23     	sw	a0, -0x24(s0)
80000206: faa00513     	li	a0, -0x56
8000020a: fca40d23     	sb	a0, -0x26(s0)
8000020e: 756d         	lui	a0, 0xffffb
80000210: aaa50513     	addi	a0, a0, -0x556
80000214: fca41c23     	sh	a0, -0x28(s0)
80000218: aaaab537     	lui	a0, 0xaaaab
8000021c: aaa50513     	addi	a0, a0, -0x556
80000220: fca42a23     	sw	a0, -0x2c(s0)
80000224: fe440513     	addi	a0, s0, -0x1c
80000228: fd440593     	addi	a1, s0, -0x2c
8000022c: 461d         	li	a2, 0x7
8000022e: 5ee000ef     	jal	0x8000081c <mf_relu_i8>
80000232: fc843503     	ld	a0, -0x38(s0)
80000236: fca42823     	sw	a0, -0x30(s0)
8000023a: a009         	j	0x8000023c <test_relu_i8+0x70>
8000023c: fd042583     	lw	a1, -0x30(s0)
80000240: 4519         	li	a0, 0x6
80000242: 02b54d63     	blt	a0, a1, 0x8000027c <test_relu_i8+0xb0>
80000246: a009         	j	0x80000248 <test_relu_i8+0x7c>
80000248: fd042603     	lw	a2, -0x30(s0)
8000024c: fd440513     	addi	a0, s0, -0x2c
80000250: 9532         	add	a0, a0, a2
80000252: 00050503     	lb	a0, 0x0(a0)
80000256: fdc40593     	addi	a1, s0, -0x24
8000025a: 95b2         	add	a1, a1, a2
8000025c: 00058583     	lb	a1, 0x0(a1)
80000260: 00b50763     	beq	a0, a1, 0x8000026e <test_relu_i8+0xa2>
80000264: a009         	j	0x80000266 <test_relu_i8+0x9a>
80000266: 4505         	li	a0, 0x1
80000268: fea42623     	sw	a0, -0x14(s0)
8000026c: a821         	j	0x80000284 <test_relu_i8+0xb8>
8000026e: a009         	j	0x80000270 <test_relu_i8+0xa4>
80000270: fd042503     	lw	a0, -0x30(s0)
80000274: 2505         	addiw	a0, a0, 0x1
80000276: fca42823     	sw	a0, -0x30(s0)
8000027a: b7c9         	j	0x8000023c <test_relu_i8+0x70>
8000027c: 4501         	li	a0, 0x0
8000027e: fea42623     	sw	a0, -0x14(s0)
80000282: a009         	j	0x80000284 <test_relu_i8+0xb8>
80000284: fec42503     	lw	a0, -0x14(s0)
80000288: fc040113     	addi	sp, s0, -0x40
8000028c: 70e2         	ld	ra, 0x38(sp)
8000028e: 7442         	ld	s0, 0x30(sp)
80000290: 6121         	addi	sp, sp, 0x40
80000292: 8082         	ret

0000000080000294 <test_relu_i16>:
; test_relu_i16():
80000294: 711d         	addi	sp, sp, -0x60
80000296: ec86         	sd	ra, 0x58(sp)
80000298: e8a2         	sd	s0, 0x50(sp)
8000029a: 1080         	addi	s0, sp, 0x60
8000029c: 00001517     	auipc	a0, 0x1
800002a0: fcc50513     	addi	a0, a0, -0x34
800002a4: c8847057     	vsetivli	zero, 0x8, e16, m1, tu, ma
800002a8: 02055407     	vle16.v	v8, (a0)
800002ac: fd040513     	addi	a0, s0, -0x30
800002b0: 02055427     	vse16.v	v8, (a0)
800002b4: 00001597     	auipc	a1, 0x1
800002b8: ff458593     	addi	a1, a1, -0xc
800002bc: 0205d407     	vle16.v	v8, (a1)
800002c0: fc040593     	addi	a1, s0, -0x40
800002c4: 0205d427     	vse16.v	v8, (a1)
800002c8: 0aa00593     	li	a1, 0xaa
800002cc: c8087057     	vsetivli	zero, 0x10, e8, m1, tu, ma
800002d0: 5e05c457     	vmv.v.x	v8, a1
800002d4: fb040593     	addi	a1, s0, -0x50
800002d8: 02058427     	vse8.v	v8, (a1)
800002dc: 4621         	li	a2, 0x8
800002de: 63a000ef     	jal	0x80000918 <mf_relu_i16>
800002e2: 4501         	li	a0, 0x0
800002e4: faa42623     	sw	a0, -0x54(s0)
800002e8: a009         	j	0x800002ea <test_relu_i16+0x56>
800002ea: fac42583     	lw	a1, -0x54(s0)
800002ee: 451d         	li	a0, 0x7
800002f0: 02b54f63     	blt	a0, a1, 0x8000032e <test_relu_i16+0x9a>
800002f4: a009         	j	0x800002f6 <test_relu_i16+0x62>
800002f6: fac42503     	lw	a0, -0x54(s0)
800002fa: 00151613     	slli	a2, a0, 0x1
800002fe: fb040513     	addi	a0, s0, -0x50
80000302: 9532         	add	a0, a0, a2
80000304: 00051503     	lh	a0, 0x0(a0)
80000308: fc040593     	addi	a1, s0, -0x40
8000030c: 95b2         	add	a1, a1, a2
8000030e: 00059583     	lh	a1, 0x0(a1)
80000312: 00b50763     	beq	a0, a1, 0x80000320 <test_relu_i16+0x8c>
80000316: a009         	j	0x80000318 <test_relu_i16+0x84>
80000318: 4505         	li	a0, 0x1
8000031a: fea42623     	sw	a0, -0x14(s0)
8000031e: a821         	j	0x80000336 <test_relu_i16+0xa2>
80000320: a009         	j	0x80000322 <test_relu_i16+0x8e>
80000322: fac42503     	lw	a0, -0x54(s0)
80000326: 2505         	addiw	a0, a0, 0x1
80000328: faa42623     	sw	a0, -0x54(s0)
8000032c: bf7d         	j	0x800002ea <test_relu_i16+0x56>
8000032e: 4501         	li	a0, 0x0
80000330: fea42623     	sw	a0, -0x14(s0)
80000334: a009         	j	0x80000336 <test_relu_i16+0xa2>
80000336: fec42503     	lw	a0, -0x14(s0)
8000033a: fa040113     	addi	sp, s0, -0x60
8000033e: 60e6         	ld	ra, 0x58(sp)
80000340: 6446         	ld	s0, 0x50(sp)
80000342: 6125         	addi	sp, sp, 0x60
80000344: 8082         	ret

0000000080000346 <test_elemwise_add_i8>:
; test_elemwise_add_i8():
80000346: 7179         	addi	sp, sp, -0x30
80000348: f406         	sd	ra, 0x28(sp)
8000034a: f022         	sd	s0, 0x20(sp)
8000034c: 1800         	addi	s0, sp, 0x30
8000034e: 04030537     	lui	a0, 0x4030
80000352: 20150513     	addi	a0, a0, 0x201
80000356: fea42423     	sw	a0, -0x18(s0)
8000035a: 281e1537     	lui	a0, 0x281e1
8000035e: 40a50513     	addi	a0, a0, 0x40a
80000362: fea42223     	sw	a0, -0x1c(s0)
80000366: 2c211537     	lui	a0, 0x2c211
8000036a: 60b50513     	addi	a0, a0, 0x60b
8000036e: fea42023     	sw	a0, -0x20(s0)
80000372: 4501         	li	a0, 0x0
80000374: fca43823     	sd	a0, -0x30(s0)
80000378: fca42e23     	sw	a0, -0x24(s0)
8000037c: fe840513     	addi	a0, s0, -0x18
80000380: fe440593     	addi	a1, s0, -0x1c
80000384: fdc40613     	addi	a2, s0, -0x24
80000388: 4691         	li	a3, 0x4
8000038a: 68e000ef     	jal	0x80000a18 <mf_elemwise_add_i8>
8000038e: fd043503     	ld	a0, -0x30(s0)
80000392: fca42c23     	sw	a0, -0x28(s0)
80000396: a009         	j	0x80000398 <test_elemwise_add_i8+0x52>
80000398: fd842583     	lw	a1, -0x28(s0)
8000039c: 450d         	li	a0, 0x3
8000039e: 02b54d63     	blt	a0, a1, 0x800003d8 <test_elemwise_add_i8+0x92>
800003a2: a009         	j	0x800003a4 <test_elemwise_add_i8+0x5e>
800003a4: fd842603     	lw	a2, -0x28(s0)
800003a8: fdc40513     	addi	a0, s0, -0x24
800003ac: 9532         	add	a0, a0, a2
800003ae: 00050503     	lb	a0, 0x0(a0)
800003b2: fe040593     	addi	a1, s0, -0x20
800003b6: 95b2         	add	a1, a1, a2
800003b8: 00058583     	lb	a1, 0x0(a1)
800003bc: 00b50763     	beq	a0, a1, 0x800003ca <test_elemwise_add_i8+0x84>
800003c0: a009         	j	0x800003c2 <test_elemwise_add_i8+0x7c>
800003c2: 4505         	li	a0, 0x1
800003c4: fea42623     	sw	a0, -0x14(s0)
800003c8: a821         	j	0x800003e0 <test_elemwise_add_i8+0x9a>
800003ca: a009         	j	0x800003cc <test_elemwise_add_i8+0x86>
800003cc: fd842503     	lw	a0, -0x28(s0)
800003d0: 2505         	addiw	a0, a0, 0x1
800003d2: fca42c23     	sw	a0, -0x28(s0)
800003d6: b7c9         	j	0x80000398 <test_elemwise_add_i8+0x52>
800003d8: 4501         	li	a0, 0x0
800003da: fea42623     	sw	a0, -0x14(s0)
800003de: a009         	j	0x800003e0 <test_elemwise_add_i8+0x9a>
800003e0: fec42503     	lw	a0, -0x14(s0)
800003e4: fd040113     	addi	sp, s0, -0x30
800003e8: 70a2         	ld	ra, 0x28(sp)
800003ea: 7402         	ld	s0, 0x20(sp)
800003ec: 6145         	addi	sp, sp, 0x30
800003ee: 8082         	ret

00000000800003f0 <test_elemwise_add_i32>:
; test_elemwise_add_i32():
800003f0: 7159         	addi	sp, sp, -0x70
800003f2: f486         	sd	ra, 0x68(sp)
800003f4: f0a2         	sd	s0, 0x60(sp)
800003f6: 1880         	addi	s0, sp, 0x70
800003f8: 00001517     	auipc	a0, 0x1
800003fc: e6050513     	addi	a0, a0, -0x1a0
80000400: c9027057     	vsetivli	zero, 0x4, e32, m1, tu, ma
80000404: 02056407     	vle32.v	v8, (a0)
80000408: fd040513     	addi	a0, s0, -0x30
8000040c: 02056427     	vse32.v	v8, (a0)
80000410: 00001597     	auipc	a1, 0x1
80000414: e6858593     	addi	a1, a1, -0x198
80000418: 0205e407     	vle32.v	v8, (a1)
8000041c: fc040593     	addi	a1, s0, -0x40
80000420: 0205e427     	vse32.v	v8, (a1)
80000424: 00001617     	auipc	a2, 0x1
80000428: e7460613     	addi	a2, a2, -0x18c
8000042c: 02066407     	vle32.v	v8, (a2)
80000430: fb040613     	addi	a2, s0, -0x50
80000434: 02066427     	vse32.v	v8, (a2)
80000438: c9817057     	vsetivli	zero, 0x2, e64, m1, tu, ma
8000043c: 5e003457     	vmv.v.i	v8, 0x0
80000440: fa040613     	addi	a2, s0, -0x60
80000444: 02067427     	vse64.v	v8, (a2)
80000448: 4691         	li	a3, 0x4
8000044a: 6d8000ef     	jal	0x80000b22 <mf_elemwise_add_i32>
8000044e: 4501         	li	a0, 0x0
80000450: f8a42e23     	sw	a0, -0x64(s0)
80000454: a009         	j	0x80000456 <test_elemwise_add_i32+0x66>
80000456: f9c42583     	lw	a1, -0x64(s0)
8000045a: 450d         	li	a0, 0x3
8000045c: 02b54d63     	blt	a0, a1, 0x80000496 <test_elemwise_add_i32+0xa6>
80000460: a009         	j	0x80000462 <test_elemwise_add_i32+0x72>
80000462: f9c42503     	lw	a0, -0x64(s0)
80000466: 00251613     	slli	a2, a0, 0x2
8000046a: fa040513     	addi	a0, s0, -0x60
8000046e: 9532         	add	a0, a0, a2
80000470: 4108         	lw	a0, 0x0(a0)
80000472: fb040593     	addi	a1, s0, -0x50
80000476: 95b2         	add	a1, a1, a2
80000478: 418c         	lw	a1, 0x0(a1)
8000047a: 00b50763     	beq	a0, a1, 0x80000488 <test_elemwise_add_i32+0x98>
8000047e: a009         	j	0x80000480 <test_elemwise_add_i32+0x90>
80000480: 4505         	li	a0, 0x1
80000482: fea42623     	sw	a0, -0x14(s0)
80000486: a821         	j	0x8000049e <test_elemwise_add_i32+0xae>
80000488: a009         	j	0x8000048a <test_elemwise_add_i32+0x9a>
8000048a: f9c42503     	lw	a0, -0x64(s0)
8000048e: 2505         	addiw	a0, a0, 0x1
80000490: f8a42e23     	sw	a0, -0x64(s0)
80000494: b7c9         	j	0x80000456 <test_elemwise_add_i32+0x66>
80000496: 4501         	li	a0, 0x0
80000498: fea42623     	sw	a0, -0x14(s0)
8000049c: a009         	j	0x8000049e <test_elemwise_add_i32+0xae>
8000049e: fec42503     	lw	a0, -0x14(s0)
800004a2: f9040113     	addi	sp, s0, -0x70
800004a6: 70a6         	ld	ra, 0x68(sp)
800004a8: 7406         	ld	s0, 0x60(sp)
800004aa: 6165         	addi	sp, sp, 0x70
800004ac: 8082         	ret

00000000800004ae <test_elemwise_mul_i8>:
; test_elemwise_mul_i8():
800004ae: 7179         	addi	sp, sp, -0x30
800004b0: f406         	sd	ra, 0x28(sp)
800004b2: f022         	sd	s0, 0x20(sp)
800004b4: 1800         	addi	s0, sp, 0x30
800004b6: 05040537     	lui	a0, 0x5040
800004ba: 30250513     	addi	a0, a0, 0x302
800004be: fea42423     	sw	a0, -0x18(s0)
800004c2: 06050537     	lui	a0, 0x6050
800004c6: 40350513     	addi	a0, a0, 0x403
800004ca: fea42223     	sw	a0, -0x1c(s0)
800004ce: 1e141537     	lui	a0, 0x1e141
800004d2: c0650513     	addi	a0, a0, -0x3fa
800004d6: fea42023     	sw	a0, -0x20(s0)
800004da: 4501         	li	a0, 0x0
800004dc: fca43823     	sd	a0, -0x30(s0)
800004e0: fca42e23     	sw	a0, -0x24(s0)
800004e4: fe840513     	addi	a0, s0, -0x18
800004e8: fe440593     	addi	a1, s0, -0x1c
800004ec: fdc40613     	addi	a2, s0, -0x24
800004f0: 4691         	li	a3, 0x4
800004f2: 740000ef     	jal	0x80000c32 <mf_elemwise_mul_i8>
800004f6: fd043503     	ld	a0, -0x30(s0)
800004fa: fca42c23     	sw	a0, -0x28(s0)
800004fe: a009         	j	0x80000500 <test_elemwise_mul_i8+0x52>
80000500: fd842583     	lw	a1, -0x28(s0)
80000504: 450d         	li	a0, 0x3
80000506: 02b54d63     	blt	a0, a1, 0x80000540 <test_elemwise_mul_i8+0x92>
8000050a: a009         	j	0x8000050c <test_elemwise_mul_i8+0x5e>
8000050c: fd842603     	lw	a2, -0x28(s0)
80000510: fdc40513     	addi	a0, s0, -0x24
80000514: 9532         	add	a0, a0, a2
80000516: 00050503     	lb	a0, 0x0(a0)
8000051a: fe040593     	addi	a1, s0, -0x20
8000051e: 95b2         	add	a1, a1, a2
80000520: 00058583     	lb	a1, 0x0(a1)
80000524: 00b50763     	beq	a0, a1, 0x80000532 <test_elemwise_mul_i8+0x84>
80000528: a009         	j	0x8000052a <test_elemwise_mul_i8+0x7c>
8000052a: 4505         	li	a0, 0x1
8000052c: fea42623     	sw	a0, -0x14(s0)
80000530: a821         	j	0x80000548 <test_elemwise_mul_i8+0x9a>
80000532: a009         	j	0x80000534 <test_elemwise_mul_i8+0x86>
80000534: fd842503     	lw	a0, -0x28(s0)
80000538: 2505         	addiw	a0, a0, 0x1
8000053a: fca42c23     	sw	a0, -0x28(s0)
8000053e: b7c9         	j	0x80000500 <test_elemwise_mul_i8+0x52>
80000540: 4501         	li	a0, 0x0
80000542: fea42623     	sw	a0, -0x14(s0)
80000546: a009         	j	0x80000548 <test_elemwise_mul_i8+0x9a>
80000548: fec42503     	lw	a0, -0x14(s0)
8000054c: fd040113     	addi	sp, s0, -0x30
80000550: 70a2         	ld	ra, 0x28(sp)
80000552: 7402         	ld	s0, 0x20(sp)
80000554: 6145         	addi	sp, sp, 0x30
80000556: 8082         	ret

0000000080000558 <test_elemwise_mul_scalar_i16>:
; test_elemwise_mul_scalar_i16():
80000558: 715d         	addi	sp, sp, -0x50
8000055a: e486         	sd	ra, 0x48(sp)
8000055c: e0a2         	sd	s0, 0x40(sp)
8000055e: 0880         	addi	s0, sp, 0x50
80000560: 40003537     	lui	a0, 0x40003
80000564: 050e         	slli	a0, a0, 0x3
80000566: 0505         	addi	a0, a0, 0x1
80000568: 0546         	slli	a0, a0, 0x11
8000056a: 0505         	addi	a0, a0, 0x1
8000056c: fea43023     	sd	a0, -0x20(s0)
80000570: 4515         	li	a0, 0x5
80000572: fca41f23     	sh	a0, -0x22(s0)
80000576: 00001517     	auipc	a0, 0x1
8000057a: caa50513     	addi	a0, a0, -0x356
8000057e: 6108         	ld	a0, 0x0(a0)
80000580: fca43823     	sd	a0, -0x30(s0)
80000584: 4501         	li	a0, 0x0
80000586: faa43c23     	sd	a0, -0x48(s0)
8000058a: fca43423     	sd	a0, -0x38(s0)
8000058e: fde41583     	lh	a1, -0x22(s0)
80000592: fe040513     	addi	a0, s0, -0x20
80000596: fc840613     	addi	a2, s0, -0x38
8000059a: 4691         	li	a3, 0x4
8000059c: 7a0000ef     	jal	0x80000d3c <mf_elemwise_mul_scalar_i16>
800005a0: fb843503     	ld	a0, -0x48(s0)
800005a4: fca42223     	sw	a0, -0x3c(s0)
800005a8: a009         	j	0x800005aa <test_elemwise_mul_scalar_i16+0x52>
800005aa: fc442583     	lw	a1, -0x3c(s0)
800005ae: 450d         	li	a0, 0x3
800005b0: 02b54f63     	blt	a0, a1, 0x800005ee <test_elemwise_mul_scalar_i16+0x96>
800005b4: a009         	j	0x800005b6 <test_elemwise_mul_scalar_i16+0x5e>
800005b6: fc442503     	lw	a0, -0x3c(s0)
800005ba: 00151613     	slli	a2, a0, 0x1
800005be: fc840513     	addi	a0, s0, -0x38
800005c2: 9532         	add	a0, a0, a2
800005c4: 00051503     	lh	a0, 0x0(a0)
800005c8: fd040593     	addi	a1, s0, -0x30
800005cc: 95b2         	add	a1, a1, a2
800005ce: 00059583     	lh	a1, 0x0(a1)
800005d2: 00b50763     	beq	a0, a1, 0x800005e0 <test_elemwise_mul_scalar_i16+0x88>
800005d6: a009         	j	0x800005d8 <test_elemwise_mul_scalar_i16+0x80>
800005d8: 4505         	li	a0, 0x1
800005da: fea42623     	sw	a0, -0x14(s0)
800005de: a821         	j	0x800005f6 <test_elemwise_mul_scalar_i16+0x9e>
800005e0: a009         	j	0x800005e2 <test_elemwise_mul_scalar_i16+0x8a>
800005e2: fc442503     	lw	a0, -0x3c(s0)
800005e6: 2505         	addiw	a0, a0, 0x1
800005e8: fca42223     	sw	a0, -0x3c(s0)
800005ec: bf7d         	j	0x800005aa <test_elemwise_mul_scalar_i16+0x52>
800005ee: 4501         	li	a0, 0x0
800005f0: fea42623     	sw	a0, -0x14(s0)
800005f4: a009         	j	0x800005f6 <test_elemwise_mul_scalar_i16+0x9e>
800005f6: fec42503     	lw	a0, -0x14(s0)
800005fa: fb040113     	addi	sp, s0, -0x50
800005fe: 60a6         	ld	ra, 0x48(sp)
80000600: 6406         	ld	s0, 0x40(sp)
80000602: 6161         	addi	sp, sp, 0x50
80000604: 8082         	ret

0000000080000606 <test_quantize_i32_to_i8>:
; test_quantize_i32_to_i8():
80000606: 715d         	addi	sp, sp, -0x50
80000608: e486         	sd	ra, 0x48(sp)
8000060a: e0a2         	sd	s0, 0x40(sp)
8000060c: 0880         	addi	s0, sp, 0x50
8000060e: 00001517     	auipc	a0, 0x1
80000612: c7a50513     	addi	a0, a0, -0x386
80000616: c9027057     	vsetivli	zero, 0x4, e32, m1, tu, ma
8000061a: 02056407     	vle32.v	v8, (a0)
8000061e: fd040513     	addi	a0, s0, -0x30
80000622: 02056427     	vse32.v	v8, (a0)
80000626: 04ff05b7     	lui	a1, 0x4ff0
8000062a: 20158593     	addi	a1, a1, 0x201
8000062e: fcb42623     	sw	a1, -0x34(s0)
80000632: 4701         	li	a4, 0x0
80000634: fae43c23     	sd	a4, -0x48(s0)
80000638: fce42423     	sw	a4, -0x38(s0)
8000063c: fc840593     	addi	a1, s0, -0x38
80000640: 4611         	li	a2, 0x4
80000642: 46a1         	li	a3, 0x8
80000644: 7d2000ef     	jal	0x80000e16 <mf_quantize_i32_to_i8>
80000648: fb843503     	ld	a0, -0x48(s0)
8000064c: fca42223     	sw	a0, -0x3c(s0)
80000650: a009         	j	0x80000652 <test_quantize_i32_to_i8+0x4c>
80000652: fc442583     	lw	a1, -0x3c(s0)
80000656: 450d         	li	a0, 0x3
80000658: 02b54d63     	blt	a0, a1, 0x80000692 <test_quantize_i32_to_i8+0x8c>
8000065c: a009         	j	0x8000065e <test_quantize_i32_to_i8+0x58>
8000065e: fc442603     	lw	a2, -0x3c(s0)
80000662: fc840513     	addi	a0, s0, -0x38
80000666: 9532         	add	a0, a0, a2
80000668: 00050503     	lb	a0, 0x0(a0)
8000066c: fcc40593     	addi	a1, s0, -0x34
80000670: 95b2         	add	a1, a1, a2
80000672: 00058583     	lb	a1, 0x0(a1)
80000676: 00b50763     	beq	a0, a1, 0x80000684 <test_quantize_i32_to_i8+0x7e>
8000067a: a009         	j	0x8000067c <test_quantize_i32_to_i8+0x76>
8000067c: 4505         	li	a0, 0x1
8000067e: fea42623     	sw	a0, -0x14(s0)
80000682: a821         	j	0x8000069a <test_quantize_i32_to_i8+0x94>
80000684: a009         	j	0x80000686 <test_quantize_i32_to_i8+0x80>
80000686: fc442503     	lw	a0, -0x3c(s0)
8000068a: 2505         	addiw	a0, a0, 0x1
8000068c: fca42223     	sw	a0, -0x3c(s0)
80000690: b7c9         	j	0x80000652 <test_quantize_i32_to_i8+0x4c>
80000692: 4501         	li	a0, 0x0
80000694: fea42623     	sw	a0, -0x14(s0)
80000698: a009         	j	0x8000069a <test_quantize_i32_to_i8+0x94>
8000069a: fec42503     	lw	a0, -0x14(s0)
8000069e: fb040113     	addi	sp, s0, -0x50
800006a2: 60a6         	ld	ra, 0x48(sp)
800006a4: 6406         	ld	s0, 0x40(sp)
800006a6: 6161         	addi	sp, sp, 0x50
800006a8: 8082         	ret

00000000800006aa <test_dequantize_i8_to_i32>:
; test_dequantize_i8_to_i32():
800006aa: 715d         	addi	sp, sp, -0x50
800006ac: e486         	sd	ra, 0x48(sp)
800006ae: e0a2         	sd	s0, 0x40(sp)
800006b0: 0880         	addi	s0, sp, 0x50
800006b2: 04ff0537     	lui	a0, 0x4ff0
800006b6: 20150513     	addi	a0, a0, 0x201
800006ba: fea42423     	sw	a0, -0x18(s0)
800006be: 00001517     	auipc	a0, 0x1
800006c2: b8a50513     	addi	a0, a0, -0x476
800006c6: c9027057     	vsetivli	zero, 0x4, e32, m1, tu, ma
800006ca: 02056407     	vle32.v	v8, (a0)
800006ce: fd040513     	addi	a0, s0, -0x30
800006d2: 02056427     	vse32.v	v8, (a0)
800006d6: c9817057     	vsetivli	zero, 0x2, e64, m1, tu, ma
800006da: 5e003457     	vmv.v.i	v8, 0x0
800006de: fc040593     	addi	a1, s0, -0x40
800006e2: 0205f427     	vse64.v	v8, (a1)
800006e6: fe840513     	addi	a0, s0, -0x18
800006ea: 4611         	li	a2, 0x4
800006ec: 06400693     	li	a3, 0x64
800006f0: 0cf000ef     	jal	0x80000fbe <mf_dequantize_i8_to_i32>
800006f4: 4501         	li	a0, 0x0
800006f6: faa42e23     	sw	a0, -0x44(s0)
800006fa: a009         	j	0x800006fc <test_dequantize_i8_to_i32+0x52>
800006fc: fbc42583     	lw	a1, -0x44(s0)
80000700: 450d         	li	a0, 0x3
80000702: 02b54d63     	blt	a0, a1, 0x8000073c <test_dequantize_i8_to_i32+0x92>
80000706: a009         	j	0x80000708 <test_dequantize_i8_to_i32+0x5e>
80000708: fbc42503     	lw	a0, -0x44(s0)
8000070c: 00251613     	slli	a2, a0, 0x2
80000710: fc040513     	addi	a0, s0, -0x40
80000714: 9532         	add	a0, a0, a2
80000716: 4108         	lw	a0, 0x0(a0)
80000718: fd040593     	addi	a1, s0, -0x30
8000071c: 95b2         	add	a1, a1, a2
8000071e: 418c         	lw	a1, 0x0(a1)
80000720: 00b50763     	beq	a0, a1, 0x8000072e <test_dequantize_i8_to_i32+0x84>
80000724: a009         	j	0x80000726 <test_dequantize_i8_to_i32+0x7c>
80000726: 4505         	li	a0, 0x1
80000728: fea42623     	sw	a0, -0x14(s0)
8000072c: a821         	j	0x80000744 <test_dequantize_i8_to_i32+0x9a>
8000072e: a009         	j	0x80000730 <test_dequantize_i8_to_i32+0x86>
80000730: fbc42503     	lw	a0, -0x44(s0)
80000734: 2505         	addiw	a0, a0, 0x1
80000736: faa42e23     	sw	a0, -0x44(s0)
8000073a: b7c9         	j	0x800006fc <test_dequantize_i8_to_i32+0x52>
8000073c: 4501         	li	a0, 0x0
8000073e: fea42623     	sw	a0, -0x14(s0)
80000742: a009         	j	0x80000744 <test_dequantize_i8_to_i32+0x9a>
80000744: fec42503     	lw	a0, -0x14(s0)
80000748: fb040113     	addi	sp, s0, -0x50
8000074c: 60a6         	ld	ra, 0x48(sp)
8000074e: 6406         	ld	s0, 0x40(sp)
80000750: 6161         	addi	sp, sp, 0x50
80000752: 8082         	ret

0000000080000754 <test_bias_add>:
; test_bias_add():
80000754: 7119         	addi	sp, sp, -0x80
80000756: fc86         	sd	ra, 0x78(sp)
80000758: f8a2         	sd	s0, 0x70(sp)
8000075a: 0100         	addi	s0, sp, 0x80
8000075c: 00001517     	auipc	a0, 0x1
80000760: b7850593     	addi	a1, a0, -0x488
80000764: c9027057     	vsetivli	zero, 0x4, e32, m1, tu, ma
80000768: 0205e407     	vle32.v	v8, (a1)
8000076c: fc040513     	addi	a0, s0, -0x40
80000770: 02056427     	vse32.v	v8, (a0)
80000774: 05c1         	addi	a1, a1, 0x10
80000776: 0205e407     	vle32.v	v8, (a1)
8000077a: fd040593     	addi	a1, s0, -0x30
8000077e: 0205e427     	vse32.v	v8, (a1)
80000782: 4595         	li	a1, 0x5
80000784: 158a         	slli	a1, a1, 0x22
80000786: 05a9         	addi	a1, a1, 0xa
80000788: fab43c23     	sd	a1, -0x48(s0)
8000078c: 00001597     	auipc	a1, 0x1
80000790: b6858593     	addi	a1, a1, -0x498
80000794: 01058613     	addi	a2, a1, 0x10
80000798: 02066407     	vle32.v	v8, (a2)
8000079c: fa040613     	addi	a2, s0, -0x60
800007a0: 02066427     	vse32.v	v8, (a2)
800007a4: 0205e407     	vle32.v	v8, (a1)
800007a8: f9040593     	addi	a1, s0, -0x70
800007ac: 0205e427     	vse32.v	v8, (a1)
800007b0: fb840593     	addi	a1, s0, -0x48
800007b4: 4609         	li	a2, 0x2
800007b6: 4691         	li	a3, 0x4
800007b8: 13d000ef     	jal	0x800010f4 <mf_bias_add_i32>
800007bc: 4501         	li	a0, 0x0
800007be: f8a42623     	sw	a0, -0x74(s0)
800007c2: a009         	j	0x800007c4 <test_bias_add+0x70>
800007c4: f8c42583     	lw	a1, -0x74(s0)
800007c8: 451d         	li	a0, 0x7
800007ca: 02b54d63     	blt	a0, a1, 0x80000804 <test_bias_add+0xb0>
800007ce: a009         	j	0x800007d0 <test_bias_add+0x7c>
800007d0: f8c42503     	lw	a0, -0x74(s0)
800007d4: 00251613     	slli	a2, a0, 0x2
800007d8: fc040513     	addi	a0, s0, -0x40
800007dc: 9532         	add	a0, a0, a2
800007de: 4108         	lw	a0, 0x0(a0)
800007e0: f9040593     	addi	a1, s0, -0x70
800007e4: 95b2         	add	a1, a1, a2
800007e6: 418c         	lw	a1, 0x0(a1)
800007e8: 00b50763     	beq	a0, a1, 0x800007f6 <test_bias_add+0xa2>
800007ec: a009         	j	0x800007ee <test_bias_add+0x9a>
800007ee: 4505         	li	a0, 0x1
800007f0: fea42623     	sw	a0, -0x14(s0)
800007f4: a821         	j	0x8000080c <test_bias_add+0xb8>
800007f6: a009         	j	0x800007f8 <test_bias_add+0xa4>
800007f8: f8c42503     	lw	a0, -0x74(s0)
800007fc: 2505         	addiw	a0, a0, 0x1
800007fe: f8a42623     	sw	a0, -0x74(s0)
80000802: b7c9         	j	0x800007c4 <test_bias_add+0x70>
80000804: 4501         	li	a0, 0x0
80000806: fea42623     	sw	a0, -0x14(s0)
8000080a: a009         	j	0x8000080c <test_bias_add+0xb8>
8000080c: fec42503     	lw	a0, -0x14(s0)
80000810: f8040113     	addi	sp, s0, -0x80
80000814: 70e6         	ld	ra, 0x78(sp)
80000816: 7446         	ld	s0, 0x70(sp)
80000818: 6109         	addi	sp, sp, 0x80
8000081a: 8082         	ret

000000008000081c <mf_relu_i8>:
; mf_relu_i8():
8000081c: 7139         	addi	sp, sp, -0x40
8000081e: fc06         	sd	ra, 0x38(sp)
80000820: f822         	sd	s0, 0x30(sp)
80000822: 0080         	addi	s0, sp, 0x40
80000824: c22026f3     	csrr	a3, vlenb
80000828: 4731         	li	a4, 0xc
8000082a: 02e686b3     	mul	a3, a3, a4
8000082e: 40d10133     	sub	sp, sp, a3
80000832: fea43023     	sd	a0, -0x20(s0)
80000836: fcb43c23     	sd	a1, -0x28(s0)
8000083a: fcc42a23     	sw	a2, -0x2c(s0)
8000083e: 4501         	li	a0, 0x0
80000840: fca42823     	sw	a0, -0x30(s0)
80000844: a009         	j	0x80000846 <mf_relu_i8+0x2a>
80000846: fd042503     	lw	a0, -0x30(s0)
8000084a: fd442583     	lw	a1, -0x2c(s0)
8000084e: 0ab55f63     	bge	a0, a1, 0x8000090c <mf_relu_i8+0xf0>
80000852: a009         	j	0x80000854 <mf_relu_i8+0x38>
80000854: fd442503     	lw	a0, -0x2c(s0)
80000858: fd042583     	lw	a1, -0x30(s0)
8000085c: 9d0d         	subw	a0, a0, a1
8000085e: 0c257557     	vsetvli	a0, a0, e8, m4, ta, ma
80000862: fca43423     	sd	a0, -0x38(s0)
80000866: fe043503     	ld	a0, -0x20(s0)
8000086a: fd042583     	lw	a1, -0x30(s0)
8000086e: 952e         	add	a0, a0, a1
80000870: fc843583     	ld	a1, -0x38(s0)
80000874: 0825f057     	vsetvli	zero, a1, e8, m4, tu, ma
80000878: 02050407     	vle8.v	v8, (a0)
8000087c: c22025f3     	csrr	a1, vlenb
80000880: 058a         	slli	a1, a1, 0x2
80000882: 40b405b3     	sub	a1, s0, a1
80000886: fc058593     	addi	a1, a1, -0x40
8000088a: 0c207557     	vsetvli	a0, zero, e8, m4, ta, ma
8000088e: 02058427     	vse8.v	v8, (a1)
80000892: fc843503     	ld	a0, -0x38(s0)
80000896: 08257057     	vsetvli	zero, a0, e8, m4, tu, ma
8000089a: 5e003457     	vmv.v.i	v8, 0x0
8000089e: c2202573     	csrr	a0, vlenb
800008a2: 050e         	slli	a0, a0, 0x3
800008a4: 40a40533     	sub	a0, s0, a0
800008a8: fc050513     	addi	a0, a0, -0x40
800008ac: 0c207657     	vsetvli	a2, zero, e8, m4, ta, ma
800008b0: 02050427     	vse8.v	v8, (a0)
800008b4: 02058607     	vle8.v	v12, (a1)
800008b8: 02050807     	vle8.v	v16, (a0)
800008bc: fc843503     	ld	a0, -0x38(s0)
800008c0: 08257057     	vsetvli	zero, a0, e8, m4, tu, ma
800008c4: 1ec80457     	vmax.vv	v8, v12, v16
800008c8: c22025f3     	csrr	a1, vlenb
800008cc: 4531         	li	a0, 0xc
800008ce: 02a585b3     	mul	a1, a1, a0
800008d2: 40b405b3     	sub	a1, s0, a1
800008d6: fc058593     	addi	a1, a1, -0x40
800008da: 0c207557     	vsetvli	a0, zero, e8, m4, ta, ma
800008de: 02058427     	vse8.v	v8, (a1)
800008e2: fd843503     	ld	a0, -0x28(s0)
800008e6: fd042603     	lw	a2, -0x30(s0)
800008ea: 9532         	add	a0, a0, a2
800008ec: 02058407     	vle8.v	v8, (a1)
800008f0: fc843583     	ld	a1, -0x38(s0)
800008f4: 0c25f057     	vsetvli	zero, a1, e8, m4, ta, ma
800008f8: 02050427     	vse8.v	v8, (a0)
800008fc: fc843583     	ld	a1, -0x38(s0)
80000900: fd042503     	lw	a0, -0x30(s0)
80000904: 9d2d         	addw	a0, a0, a1
80000906: fca42823     	sw	a0, -0x30(s0)
8000090a: bf35         	j	0x80000846 <mf_relu_i8+0x2a>
8000090c: fc040113     	addi	sp, s0, -0x40
80000910: 70e2         	ld	ra, 0x38(sp)
80000912: 7442         	ld	s0, 0x30(sp)
80000914: 6121         	addi	sp, sp, 0x40
80000916: 8082         	ret

0000000080000918 <mf_relu_i16>:
; mf_relu_i16():
80000918: 7139         	addi	sp, sp, -0x40
8000091a: fc06         	sd	ra, 0x38(sp)
8000091c: f822         	sd	s0, 0x30(sp)
8000091e: 0080         	addi	s0, sp, 0x40
80000920: c22026f3     	csrr	a3, vlenb
80000924: 4731         	li	a4, 0xc
80000926: 02e686b3     	mul	a3, a3, a4
8000092a: 40d10133     	sub	sp, sp, a3
8000092e: fea43023     	sd	a0, -0x20(s0)
80000932: fcb43c23     	sd	a1, -0x28(s0)
80000936: fcc42a23     	sw	a2, -0x2c(s0)
8000093a: 4501         	li	a0, 0x0
8000093c: fca42823     	sw	a0, -0x30(s0)
80000940: a009         	j	0x80000942 <mf_relu_i16+0x2a>
80000942: fd042503     	lw	a0, -0x30(s0)
80000946: fd442583     	lw	a1, -0x2c(s0)
8000094a: 0cb55163     	bge	a0, a1, 0x80000a0c <mf_relu_i16+0xf4>
8000094e: a009         	j	0x80000950 <mf_relu_i16+0x38>
80000950: fd442503     	lw	a0, -0x2c(s0)
80000954: fd042583     	lw	a1, -0x30(s0)
80000958: 9d0d         	subw	a0, a0, a1
8000095a: 0ca57557     	vsetvli	a0, a0, e16, m4, ta, ma
8000095e: fca43423     	sd	a0, -0x38(s0)
80000962: fe043503     	ld	a0, -0x20(s0)
80000966: fd042583     	lw	a1, -0x30(s0)
8000096a: 0586         	slli	a1, a1, 0x1
8000096c: 952e         	add	a0, a0, a1
8000096e: fc843583     	ld	a1, -0x38(s0)
80000972: 08a5f057     	vsetvli	zero, a1, e16, m4, tu, ma
80000976: 02055407     	vle16.v	v8, (a0)
8000097a: c22025f3     	csrr	a1, vlenb
8000097e: 058a         	slli	a1, a1, 0x2
80000980: 40b405b3     	sub	a1, s0, a1
80000984: fc058593     	addi	a1, a1, -0x40
80000988: 0ca07557     	vsetvli	a0, zero, e16, m4, ta, ma
8000098c: 0205d427     	vse16.v	v8, (a1)
80000990: fc843503     	ld	a0, -0x38(s0)
80000994: 08a57057     	vsetvli	zero, a0, e16, m4, tu, ma
80000998: 5e003457     	vmv.v.i	v8, 0x0
8000099c: c2202573     	csrr	a0, vlenb
800009a0: 050e         	slli	a0, a0, 0x3
800009a2: 40a40533     	sub	a0, s0, a0
800009a6: fc050513     	addi	a0, a0, -0x40
800009aa: 0ca07657     	vsetvli	a2, zero, e16, m4, ta, ma
800009ae: 02055427     	vse16.v	v8, (a0)
800009b2: 0205d607     	vle16.v	v12, (a1)
800009b6: 02055807     	vle16.v	v16, (a0)
800009ba: fc843503     	ld	a0, -0x38(s0)
800009be: 08a57057     	vsetvli	zero, a0, e16, m4, tu, ma
800009c2: 1ec80457     	vmax.vv	v8, v12, v16
800009c6: c22025f3     	csrr	a1, vlenb
800009ca: 4531         	li	a0, 0xc
800009cc: 02a585b3     	mul	a1, a1, a0
800009d0: 40b405b3     	sub	a1, s0, a1
800009d4: fc058593     	addi	a1, a1, -0x40
800009d8: 0ca07557     	vsetvli	a0, zero, e16, m4, ta, ma
800009dc: 0205d427     	vse16.v	v8, (a1)
800009e0: fd843503     	ld	a0, -0x28(s0)
800009e4: fd042603     	lw	a2, -0x30(s0)
800009e8: 0606         	slli	a2, a2, 0x1
800009ea: 9532         	add	a0, a0, a2
800009ec: 0205d407     	vle16.v	v8, (a1)
800009f0: fc843583     	ld	a1, -0x38(s0)
800009f4: 0ca5f057     	vsetvli	zero, a1, e16, m4, ta, ma
800009f8: 02055427     	vse16.v	v8, (a0)
800009fc: fc843583     	ld	a1, -0x38(s0)
80000a00: fd042503     	lw	a0, -0x30(s0)
80000a04: 9d2d         	addw	a0, a0, a1
80000a06: fca42823     	sw	a0, -0x30(s0)
80000a0a: bf25         	j	0x80000942 <mf_relu_i16+0x2a>
80000a0c: fc040113     	addi	sp, s0, -0x40
80000a10: 70e2         	ld	ra, 0x38(sp)
80000a12: 7442         	ld	s0, 0x30(sp)
80000a14: 6121         	addi	sp, sp, 0x40
80000a16: 8082         	ret

0000000080000a18 <mf_elemwise_add_i8>:
; mf_elemwise_add_i8():
80000a18: 7139         	addi	sp, sp, -0x40
80000a1a: fc06         	sd	ra, 0x38(sp)
80000a1c: f822         	sd	s0, 0x30(sp)
80000a1e: 0080         	addi	s0, sp, 0x40
80000a20: c2202773     	csrr	a4, vlenb
80000a24: 47b1         	li	a5, 0xc
80000a26: 02f70733     	mul	a4, a4, a5
80000a2a: 40e10133     	sub	sp, sp, a4
80000a2e: fea43023     	sd	a0, -0x20(s0)
80000a32: fcb43c23     	sd	a1, -0x28(s0)
80000a36: fcc43823     	sd	a2, -0x30(s0)
80000a3a: fcd42623     	sw	a3, -0x34(s0)
80000a3e: 4501         	li	a0, 0x0
80000a40: fca42423     	sw	a0, -0x38(s0)
80000a44: a009         	j	0x80000a46 <mf_elemwise_add_i8+0x2e>
80000a46: fc842503     	lw	a0, -0x38(s0)
80000a4a: fcc42583     	lw	a1, -0x34(s0)
80000a4e: 0cb55463     	bge	a0, a1, 0x80000b16 <mf_elemwise_add_i8+0xfe>
80000a52: a009         	j	0x80000a54 <mf_elemwise_add_i8+0x3c>
80000a54: fcc42503     	lw	a0, -0x34(s0)
80000a58: fc842583     	lw	a1, -0x38(s0)
80000a5c: 9d0d         	subw	a0, a0, a1
80000a5e: 0c257557     	vsetvli	a0, a0, e8, m4, ta, ma
80000a62: fca43023     	sd	a0, -0x40(s0)
80000a66: fe043503     	ld	a0, -0x20(s0)
80000a6a: fc842583     	lw	a1, -0x38(s0)
80000a6e: 952e         	add	a0, a0, a1
80000a70: fc043583     	ld	a1, -0x40(s0)
80000a74: 0825f057     	vsetvli	zero, a1, e8, m4, tu, ma
80000a78: 02050407     	vle8.v	v8, (a0)
80000a7c: c22025f3     	csrr	a1, vlenb
80000a80: 058a         	slli	a1, a1, 0x2
80000a82: 40b405b3     	sub	a1, s0, a1
80000a86: fc058593     	addi	a1, a1, -0x40
80000a8a: 0c207557     	vsetvli	a0, zero, e8, m4, ta, ma
80000a8e: 02058427     	vse8.v	v8, (a1)
80000a92: fd843503     	ld	a0, -0x28(s0)
80000a96: fc842603     	lw	a2, -0x38(s0)
80000a9a: 9532         	add	a0, a0, a2
80000a9c: fc043603     	ld	a2, -0x40(s0)
80000aa0: 08267057     	vsetvli	zero, a2, e8, m4, tu, ma
80000aa4: 02050407     	vle8.v	v8, (a0)
80000aa8: c2202573     	csrr	a0, vlenb
80000aac: 050e         	slli	a0, a0, 0x3
80000aae: 40a40533     	sub	a0, s0, a0
80000ab2: fc050513     	addi	a0, a0, -0x40
80000ab6: 0c207657     	vsetvli	a2, zero, e8, m4, ta, ma
80000aba: 02050427     	vse8.v	v8, (a0)
80000abe: 02058607     	vle8.v	v12, (a1)
80000ac2: 02050807     	vle8.v	v16, (a0)
80000ac6: fc043503     	ld	a0, -0x40(s0)
80000aca: 08257057     	vsetvli	zero, a0, e8, m4, tu, ma
80000ace: 02c80457     	vadd.vv	v8, v12, v16
80000ad2: c22025f3     	csrr	a1, vlenb
80000ad6: 4531         	li	a0, 0xc
80000ad8: 02a585b3     	mul	a1, a1, a0
80000adc: 40b405b3     	sub	a1, s0, a1
80000ae0: fc058593     	addi	a1, a1, -0x40
80000ae4: 0c207557     	vsetvli	a0, zero, e8, m4, ta, ma
80000ae8: 02058427     	vse8.v	v8, (a1)
80000aec: fd043503     	ld	a0, -0x30(s0)
80000af0: fc842603     	lw	a2, -0x38(s0)
80000af4: 9532         	add	a0, a0, a2
80000af6: 02058407     	vle8.v	v8, (a1)
80000afa: fc043583     	ld	a1, -0x40(s0)
80000afe: 0c25f057     	vsetvli	zero, a1, e8, m4, ta, ma
80000b02: 02050427     	vse8.v	v8, (a0)
80000b06: fc043583     	ld	a1, -0x40(s0)
80000b0a: fc842503     	lw	a0, -0x38(s0)
80000b0e: 9d2d         	addw	a0, a0, a1
80000b10: fca42423     	sw	a0, -0x38(s0)
80000b14: bf0d         	j	0x80000a46 <mf_elemwise_add_i8+0x2e>
80000b16: fc040113     	addi	sp, s0, -0x40
80000b1a: 70e2         	ld	ra, 0x38(sp)
80000b1c: 7442         	ld	s0, 0x30(sp)
80000b1e: 6121         	addi	sp, sp, 0x40
80000b20: 8082         	ret

0000000080000b22 <mf_elemwise_add_i32>:
; mf_elemwise_add_i32():
80000b22: 7139         	addi	sp, sp, -0x40
80000b24: fc06         	sd	ra, 0x38(sp)
80000b26: f822         	sd	s0, 0x30(sp)
80000b28: 0080         	addi	s0, sp, 0x40
80000b2a: c2202773     	csrr	a4, vlenb
80000b2e: 47b1         	li	a5, 0xc
80000b30: 02f70733     	mul	a4, a4, a5
80000b34: 40e10133     	sub	sp, sp, a4
80000b38: fea43023     	sd	a0, -0x20(s0)
80000b3c: fcb43c23     	sd	a1, -0x28(s0)
80000b40: fcc43823     	sd	a2, -0x30(s0)
80000b44: fcd42623     	sw	a3, -0x34(s0)
80000b48: 4501         	li	a0, 0x0
80000b4a: fca42423     	sw	a0, -0x38(s0)
80000b4e: a009         	j	0x80000b50 <mf_elemwise_add_i32+0x2e>
80000b50: fc842503     	lw	a0, -0x38(s0)
80000b54: fcc42583     	lw	a1, -0x34(s0)
80000b58: 0cb55763     	bge	a0, a1, 0x80000c26 <mf_elemwise_add_i32+0x104>
80000b5c: a009         	j	0x80000b5e <mf_elemwise_add_i32+0x3c>
80000b5e: fcc42503     	lw	a0, -0x34(s0)
80000b62: fc842583     	lw	a1, -0x38(s0)
80000b66: 9d0d         	subw	a0, a0, a1
80000b68: 0d257557     	vsetvli	a0, a0, e32, m4, ta, ma
80000b6c: fca43023     	sd	a0, -0x40(s0)
80000b70: fe043503     	ld	a0, -0x20(s0)
80000b74: fc842583     	lw	a1, -0x38(s0)
80000b78: 058a         	slli	a1, a1, 0x2
80000b7a: 952e         	add	a0, a0, a1
80000b7c: fc043583     	ld	a1, -0x40(s0)
80000b80: 0925f057     	vsetvli	zero, a1, e32, m4, tu, ma
80000b84: 02056407     	vle32.v	v8, (a0)
80000b88: c22025f3     	csrr	a1, vlenb
80000b8c: 058a         	slli	a1, a1, 0x2
80000b8e: 40b405b3     	sub	a1, s0, a1
80000b92: fc058593     	addi	a1, a1, -0x40
80000b96: 0d207557     	vsetvli	a0, zero, e32, m4, ta, ma
80000b9a: 0205e427     	vse32.v	v8, (a1)
80000b9e: fd843503     	ld	a0, -0x28(s0)
80000ba2: fc842603     	lw	a2, -0x38(s0)
80000ba6: 060a         	slli	a2, a2, 0x2
80000ba8: 9532         	add	a0, a0, a2
80000baa: fc043603     	ld	a2, -0x40(s0)
80000bae: 09267057     	vsetvli	zero, a2, e32, m4, tu, ma
80000bb2: 02056407     	vle32.v	v8, (a0)
80000bb6: c2202573     	csrr	a0, vlenb
80000bba: 050e         	slli	a0, a0, 0x3
80000bbc: 40a40533     	sub	a0, s0, a0
80000bc0: fc050513     	addi	a0, a0, -0x40
80000bc4: 0d207657     	vsetvli	a2, zero, e32, m4, ta, ma
80000bc8: 02056427     	vse32.v	v8, (a0)
80000bcc: 0205e607     	vle32.v	v12, (a1)
80000bd0: 02056807     	vle32.v	v16, (a0)
80000bd4: fc043503     	ld	a0, -0x40(s0)
80000bd8: 09257057     	vsetvli	zero, a0, e32, m4, tu, ma
80000bdc: 02c80457     	vadd.vv	v8, v12, v16
80000be0: c22025f3     	csrr	a1, vlenb
80000be4: 4531         	li	a0, 0xc
80000be6: 02a585b3     	mul	a1, a1, a0
80000bea: 40b405b3     	sub	a1, s0, a1
80000bee: fc058593     	addi	a1, a1, -0x40
80000bf2: 0d207557     	vsetvli	a0, zero, e32, m4, ta, ma
80000bf6: 0205e427     	vse32.v	v8, (a1)
80000bfa: fd043503     	ld	a0, -0x30(s0)
80000bfe: fc842603     	lw	a2, -0x38(s0)
80000c02: 060a         	slli	a2, a2, 0x2
80000c04: 9532         	add	a0, a0, a2
80000c06: 0205e407     	vle32.v	v8, (a1)
80000c0a: fc043583     	ld	a1, -0x40(s0)
80000c0e: 0d25f057     	vsetvli	zero, a1, e32, m4, ta, ma
80000c12: 02056427     	vse32.v	v8, (a0)
80000c16: fc043583     	ld	a1, -0x40(s0)
80000c1a: fc842503     	lw	a0, -0x38(s0)
80000c1e: 9d2d         	addw	a0, a0, a1
80000c20: fca42423     	sw	a0, -0x38(s0)
80000c24: b735         	j	0x80000b50 <mf_elemwise_add_i32+0x2e>
80000c26: fc040113     	addi	sp, s0, -0x40
80000c2a: 70e2         	ld	ra, 0x38(sp)
80000c2c: 7442         	ld	s0, 0x30(sp)
80000c2e: 6121         	addi	sp, sp, 0x40
80000c30: 8082         	ret

0000000080000c32 <mf_elemwise_mul_i8>:
; mf_elemwise_mul_i8():
80000c32: 7139         	addi	sp, sp, -0x40
80000c34: fc06         	sd	ra, 0x38(sp)
80000c36: f822         	sd	s0, 0x30(sp)
80000c38: 0080         	addi	s0, sp, 0x40
80000c3a: c2202773     	csrr	a4, vlenb
80000c3e: 47b1         	li	a5, 0xc
80000c40: 02f70733     	mul	a4, a4, a5
80000c44: 40e10133     	sub	sp, sp, a4
80000c48: fea43023     	sd	a0, -0x20(s0)
80000c4c: fcb43c23     	sd	a1, -0x28(s0)
80000c50: fcc43823     	sd	a2, -0x30(s0)
80000c54: fcd42623     	sw	a3, -0x34(s0)
80000c58: 4501         	li	a0, 0x0
80000c5a: fca42423     	sw	a0, -0x38(s0)
80000c5e: a009         	j	0x80000c60 <mf_elemwise_mul_i8+0x2e>
80000c60: fc842503     	lw	a0, -0x38(s0)
80000c64: fcc42583     	lw	a1, -0x34(s0)
80000c68: 0cb55463     	bge	a0, a1, 0x80000d30 <mf_elemwise_mul_i8+0xfe>
80000c6c: a009         	j	0x80000c6e <mf_elemwise_mul_i8+0x3c>
80000c6e: fcc42503     	lw	a0, -0x34(s0)
80000c72: fc842583     	lw	a1, -0x38(s0)
80000c76: 9d0d         	subw	a0, a0, a1
80000c78: 0c257557     	vsetvli	a0, a0, e8, m4, ta, ma
80000c7c: fca43023     	sd	a0, -0x40(s0)
80000c80: fe043503     	ld	a0, -0x20(s0)
80000c84: fc842583     	lw	a1, -0x38(s0)
80000c88: 952e         	add	a0, a0, a1
80000c8a: fc043583     	ld	a1, -0x40(s0)
80000c8e: 0825f057     	vsetvli	zero, a1, e8, m4, tu, ma
80000c92: 02050407     	vle8.v	v8, (a0)
80000c96: c22025f3     	csrr	a1, vlenb
80000c9a: 058a         	slli	a1, a1, 0x2
80000c9c: 40b405b3     	sub	a1, s0, a1
80000ca0: fc058593     	addi	a1, a1, -0x40
80000ca4: 0c207557     	vsetvli	a0, zero, e8, m4, ta, ma
80000ca8: 02058427     	vse8.v	v8, (a1)
80000cac: fd843503     	ld	a0, -0x28(s0)
80000cb0: fc842603     	lw	a2, -0x38(s0)
80000cb4: 9532         	add	a0, a0, a2
80000cb6: fc043603     	ld	a2, -0x40(s0)
80000cba: 08267057     	vsetvli	zero, a2, e8, m4, tu, ma
80000cbe: 02050407     	vle8.v	v8, (a0)
80000cc2: c2202573     	csrr	a0, vlenb
80000cc6: 050e         	slli	a0, a0, 0x3
80000cc8: 40a40533     	sub	a0, s0, a0
80000ccc: fc050513     	addi	a0, a0, -0x40
80000cd0: 0c207657     	vsetvli	a2, zero, e8, m4, ta, ma
80000cd4: 02050427     	vse8.v	v8, (a0)
80000cd8: 02058607     	vle8.v	v12, (a1)
80000cdc: 02050807     	vle8.v	v16, (a0)
80000ce0: fc043503     	ld	a0, -0x40(s0)
80000ce4: 08257057     	vsetvli	zero, a0, e8, m4, tu, ma
80000ce8: 96c82457     	vmul.vv	v8, v12, v16
80000cec: c22025f3     	csrr	a1, vlenb
80000cf0: 4531         	li	a0, 0xc
80000cf2: 02a585b3     	mul	a1, a1, a0
80000cf6: 40b405b3     	sub	a1, s0, a1
80000cfa: fc058593     	addi	a1, a1, -0x40
80000cfe: 0c207557     	vsetvli	a0, zero, e8, m4, ta, ma
80000d02: 02058427     	vse8.v	v8, (a1)
80000d06: fd043503     	ld	a0, -0x30(s0)
80000d0a: fc842603     	lw	a2, -0x38(s0)
80000d0e: 9532         	add	a0, a0, a2
80000d10: 02058407     	vle8.v	v8, (a1)
80000d14: fc043583     	ld	a1, -0x40(s0)
80000d18: 0c25f057     	vsetvli	zero, a1, e8, m4, ta, ma
80000d1c: 02050427     	vse8.v	v8, (a0)
80000d20: fc043583     	ld	a1, -0x40(s0)
80000d24: fc842503     	lw	a0, -0x38(s0)
80000d28: 9d2d         	addw	a0, a0, a1
80000d2a: fca42423     	sw	a0, -0x38(s0)
80000d2e: bf0d         	j	0x80000c60 <mf_elemwise_mul_i8+0x2e>
80000d30: fc040113     	addi	sp, s0, -0x40
80000d34: 70e2         	ld	ra, 0x38(sp)
80000d36: 7442         	ld	s0, 0x30(sp)
80000d38: 6121         	addi	sp, sp, 0x40
80000d3a: 8082         	ret

0000000080000d3c <mf_elemwise_mul_scalar_i16>:
; mf_elemwise_mul_scalar_i16():
80000d3c: 7139         	addi	sp, sp, -0x40
80000d3e: fc06         	sd	ra, 0x38(sp)
80000d40: f822         	sd	s0, 0x30(sp)
80000d42: 0080         	addi	s0, sp, 0x40
80000d44: c2202773     	csrr	a4, vlenb
80000d48: 070e         	slli	a4, a4, 0x3
80000d4a: 40e10133     	sub	sp, sp, a4
80000d4e: fea43023     	sd	a0, -0x20(s0)
80000d52: fcb41f23     	sh	a1, -0x22(s0)
80000d56: fcc43823     	sd	a2, -0x30(s0)
80000d5a: fcd42623     	sw	a3, -0x34(s0)
80000d5e: 4501         	li	a0, 0x0
80000d60: fca42423     	sw	a0, -0x38(s0)
80000d64: a009         	j	0x80000d66 <mf_elemwise_mul_scalar_i16+0x2a>
80000d66: fc842503     	lw	a0, -0x38(s0)
80000d6a: fcc42583     	lw	a1, -0x34(s0)
80000d6e: 08b55e63     	bge	a0, a1, 0x80000e0a <mf_elemwise_mul_scalar_i16+0xce>
80000d72: a009         	j	0x80000d74 <mf_elemwise_mul_scalar_i16+0x38>
80000d74: fcc42503     	lw	a0, -0x34(s0)
80000d78: fc842583     	lw	a1, -0x38(s0)
80000d7c: 9d0d         	subw	a0, a0, a1
80000d7e: 0ca57557     	vsetvli	a0, a0, e16, m4, ta, ma
80000d82: fca43023     	sd	a0, -0x40(s0)
80000d86: fe043503     	ld	a0, -0x20(s0)
80000d8a: fc842583     	lw	a1, -0x38(s0)
80000d8e: 0586         	slli	a1, a1, 0x1
80000d90: 952e         	add	a0, a0, a1
80000d92: fc043583     	ld	a1, -0x40(s0)
80000d96: 08a5f057     	vsetvli	zero, a1, e16, m4, tu, ma
80000d9a: 02055407     	vle16.v	v8, (a0)
80000d9e: c2202573     	csrr	a0, vlenb
80000da2: 050a         	slli	a0, a0, 0x2
80000da4: 40a40533     	sub	a0, s0, a0
80000da8: fc050513     	addi	a0, a0, -0x40
80000dac: 0ca075d7     	vsetvli	a1, zero, e16, m4, ta, ma
80000db0: 02055427     	vse16.v	v8, (a0)
80000db4: 02055607     	vle16.v	v12, (a0)
80000db8: fde41503     	lh	a0, -0x22(s0)
80000dbc: fc043583     	ld	a1, -0x40(s0)
80000dc0: 08a5f057     	vsetvli	zero, a1, e16, m4, tu, ma
80000dc4: 96c56457     	vmul.vx	v8, v12, a0
80000dc8: c22025f3     	csrr	a1, vlenb
80000dcc: 058e         	slli	a1, a1, 0x3
80000dce: 40b405b3     	sub	a1, s0, a1
80000dd2: fc058593     	addi	a1, a1, -0x40
80000dd6: 0ca07557     	vsetvli	a0, zero, e16, m4, ta, ma
80000dda: 0205d427     	vse16.v	v8, (a1)
80000dde: fd043503     	ld	a0, -0x30(s0)
80000de2: fc842603     	lw	a2, -0x38(s0)
80000de6: 0606         	slli	a2, a2, 0x1
80000de8: 9532         	add	a0, a0, a2
80000dea: 0205d407     	vle16.v	v8, (a1)
80000dee: fc043583     	ld	a1, -0x40(s0)
80000df2: 0ca5f057     	vsetvli	zero, a1, e16, m4, ta, ma
80000df6: 02055427     	vse16.v	v8, (a0)
80000dfa: fc043583     	ld	a1, -0x40(s0)
80000dfe: fc842503     	lw	a0, -0x38(s0)
80000e02: 9d2d         	addw	a0, a0, a1
80000e04: fca42423     	sw	a0, -0x38(s0)
80000e08: bfb9         	j	0x80000d66 <mf_elemwise_mul_scalar_i16+0x2a>
80000e0a: fc040113     	addi	sp, s0, -0x40
80000e0e: 70e2         	ld	ra, 0x38(sp)
80000e10: 7442         	ld	s0, 0x30(sp)
80000e12: 6121         	addi	sp, sp, 0x40
80000e14: 8082         	ret

0000000080000e16 <mf_quantize_i32_to_i8>:
; mf_quantize_i32_to_i8():
80000e16: 7139         	addi	sp, sp, -0x40
80000e18: fc06         	sd	ra, 0x38(sp)
80000e1a: f822         	sd	s0, 0x30(sp)
80000e1c: 0080         	addi	s0, sp, 0x40
80000e1e: c22027f3     	csrr	a5, vlenb
80000e22: 484d         	li	a6, 0x13
80000e24: 030787b3     	mul	a5, a5, a6
80000e28: 40f10133     	sub	sp, sp, a5
80000e2c: fea43023     	sd	a0, -0x20(s0)
80000e30: fcb43c23     	sd	a1, -0x28(s0)
80000e34: fcc42a23     	sw	a2, -0x2c(s0)
80000e38: fcd42823     	sw	a3, -0x30(s0)
80000e3c: fce42623     	sw	a4, -0x34(s0)
80000e40: 4501         	li	a0, 0x0
80000e42: fca42423     	sw	a0, -0x38(s0)
80000e46: a009         	j	0x80000e48 <mf_quantize_i32_to_i8+0x32>
80000e48: fc842503     	lw	a0, -0x38(s0)
80000e4c: fd442583     	lw	a1, -0x2c(s0)
80000e50: 16b55163     	bge	a0, a1, 0x80000fb2 <mf_quantize_i32_to_i8+0x19c>
80000e54: a009         	j	0x80000e56 <mf_quantize_i32_to_i8+0x40>
80000e56: fd442503     	lw	a0, -0x2c(s0)
80000e5a: fc842583     	lw	a1, -0x38(s0)
80000e5e: 9d0d         	subw	a0, a0, a1
80000e60: 0d257557     	vsetvli	a0, a0, e32, m4, ta, ma
80000e64: fca43023     	sd	a0, -0x40(s0)
80000e68: fe043503     	ld	a0, -0x20(s0)
80000e6c: fc842583     	lw	a1, -0x38(s0)
80000e70: 058a         	slli	a1, a1, 0x2
80000e72: 952e         	add	a0, a0, a1
80000e74: fc043583     	ld	a1, -0x40(s0)
80000e78: 0925f057     	vsetvli	zero, a1, e32, m4, tu, ma
80000e7c: 02056407     	vle32.v	v8, (a0)
80000e80: c2202573     	csrr	a0, vlenb
80000e84: 050a         	slli	a0, a0, 0x2
80000e86: 40a40533     	sub	a0, s0, a0
80000e8a: fc050513     	addi	a0, a0, -0x40
80000e8e: 0d2075d7     	vsetvli	a1, zero, e32, m4, ta, ma
80000e92: 02056427     	vse32.v	v8, (a0)
80000e96: 02056607     	vle32.v	v12, (a0)
80000e9a: fd042503     	lw	a0, -0x30(s0)
80000e9e: fc043583     	ld	a1, -0x40(s0)
80000ea2: 0925f057     	vsetvli	zero, a1, e32, m4, tu, ma
80000ea6: a6c54457     	vsra.vx	v8, v12, a0
80000eaa: c2202573     	csrr	a0, vlenb
80000eae: 050e         	slli	a0, a0, 0x3
80000eb0: 40a40533     	sub	a0, s0, a0
80000eb4: fc050513     	addi	a0, a0, -0x40
80000eb8: 0d2075d7     	vsetvli	a1, zero, e32, m4, ta, ma
80000ebc: 02056427     	vse32.v	v8, (a0)
80000ec0: 02056607     	vle32.v	v12, (a0)
80000ec4: fcc42503     	lw	a0, -0x34(s0)
80000ec8: fc043583     	ld	a1, -0x40(s0)
80000ecc: 0925f057     	vsetvli	zero, a1, e32, m4, tu, ma
80000ed0: 02c54457     	vadd.vx	v8, v12, a0
80000ed4: c2202573     	csrr	a0, vlenb
80000ed8: 45b1         	li	a1, 0xc
80000eda: 02b50533     	mul	a0, a0, a1
80000ede: 40a40533     	sub	a0, s0, a0
80000ee2: fc050513     	addi	a0, a0, -0x40
80000ee6: 0d2075d7     	vsetvli	a1, zero, e32, m4, ta, ma
80000eea: 02056427     	vse32.v	v8, (a0)
80000eee: 02056607     	vle32.v	v12, (a0)
80000ef2: fc043583     	ld	a1, -0x40(s0)
80000ef6: f8000513     	li	a0, -0x80
80000efa: 0925f057     	vsetvli	zero, a1, e32, m4, tu, ma
80000efe: 1ec54457     	vmax.vx	v8, v12, a0
80000f02: c2202573     	csrr	a0, vlenb
80000f06: 0512         	slli	a0, a0, 0x4
80000f08: 40a40533     	sub	a0, s0, a0
80000f0c: fc050513     	addi	a0, a0, -0x40
80000f10: 0d2075d7     	vsetvli	a1, zero, e32, m4, ta, ma
80000f14: 02056427     	vse32.v	v8, (a0)
80000f18: 02056607     	vle32.v	v12, (a0)
80000f1c: fc043603     	ld	a2, -0x40(s0)
80000f20: 07f00593     	li	a1, 0x7f
80000f24: 09267057     	vsetvli	zero, a2, e32, m4, tu, ma
80000f28: 16c5c457     	vmin.vx	v8, v12, a1
80000f2c: 0d2075d7     	vsetvli	a1, zero, e32, m4, ta, ma
80000f30: 02056427     	vse32.v	v8, (a0)
80000f34: 02056607     	vle32.v	v12, (a0)
80000f38: fc043503     	ld	a0, -0x40(s0)
80000f3c: 08957057     	vsetvli	zero, a0, e16, m2, tu, ma
80000f40: b6c03457     	vnsra.wi	v8, v12, 0x0
80000f44: c2202573     	csrr	a0, vlenb
80000f48: 45c9         	li	a1, 0x12
80000f4a: 02b50533     	mul	a0, a0, a1
80000f4e: 40a40533     	sub	a0, s0, a0
80000f52: fc050513     	addi	a0, a0, -0x40
80000f56: 0c9075d7     	vsetvli	a1, zero, e16, m2, ta, ma
80000f5a: 02055427     	vse16.v	v8, (a0)
80000f5e: 02055507     	vle16.v	v10, (a0)
80000f62: fc043503     	ld	a0, -0x40(s0)
80000f66: 08057057     	vsetvli	zero, a0, e8, m1, tu, ma
80000f6a: b6a03457     	vnsra.wi	v8, v10, 0x0
80000f6e: c22025f3     	csrr	a1, vlenb
80000f72: 454d         	li	a0, 0x13
80000f74: 02a585b3     	mul	a1, a1, a0
80000f78: 40b405b3     	sub	a1, s0, a1
80000f7c: fc058593     	addi	a1, a1, -0x40
80000f80: 0c007557     	vsetvli	a0, zero, e8, m1, ta, ma
80000f84: 02058427     	vse8.v	v8, (a1)
80000f88: fd843503     	ld	a0, -0x28(s0)
80000f8c: fc842603     	lw	a2, -0x38(s0)
80000f90: 9532         	add	a0, a0, a2
80000f92: 02058407     	vle8.v	v8, (a1)
80000f96: fc043583     	ld	a1, -0x40(s0)
80000f9a: 0c05f057     	vsetvli	zero, a1, e8, m1, ta, ma
80000f9e: 02050427     	vse8.v	v8, (a0)
80000fa2: fc043583     	ld	a1, -0x40(s0)
80000fa6: fc842503     	lw	a0, -0x38(s0)
80000faa: 9d2d         	addw	a0, a0, a1
80000fac: fca42423     	sw	a0, -0x38(s0)
80000fb0: bd61         	j	0x80000e48 <mf_quantize_i32_to_i8+0x32>
80000fb2: fc040113     	addi	sp, s0, -0x40
80000fb6: 70e2         	ld	ra, 0x38(sp)
80000fb8: 7442         	ld	s0, 0x30(sp)
80000fba: 6121         	addi	sp, sp, 0x40
80000fbc: 8082         	ret

0000000080000fbe <mf_dequantize_i8_to_i32>:
; mf_dequantize_i8_to_i32():
80000fbe: 7139         	addi	sp, sp, -0x40
80000fc0: fc06         	sd	ra, 0x38(sp)
80000fc2: f822         	sd	s0, 0x30(sp)
80000fc4: 0080         	addi	s0, sp, 0x40
80000fc6: c2202773     	csrr	a4, vlenb
80000fca: 47ad         	li	a5, 0xb
80000fcc: 02f70733     	mul	a4, a4, a5
80000fd0: 40e10133     	sub	sp, sp, a4
80000fd4: fea43023     	sd	a0, -0x20(s0)
80000fd8: fcb43c23     	sd	a1, -0x28(s0)
80000fdc: fcc42a23     	sw	a2, -0x2c(s0)
80000fe0: fcd42823     	sw	a3, -0x30(s0)
80000fe4: 4501         	li	a0, 0x0
80000fe6: fca42623     	sw	a0, -0x34(s0)
80000fea: a009         	j	0x80000fec <mf_dequantize_i8_to_i32+0x2e>
80000fec: fcc42503     	lw	a0, -0x34(s0)
80000ff0: fd442583     	lw	a1, -0x2c(s0)
80000ff4: 0eb55a63     	bge	a0, a1, 0x800010e8 <mf_dequantize_i8_to_i32+0x12a>
80000ff8: a009         	j	0x80000ffa <mf_dequantize_i8_to_i32+0x3c>
80000ffa: fd442503     	lw	a0, -0x2c(s0)
80000ffe: fcc42583     	lw	a1, -0x34(s0)
80001002: 9d0d         	subw	a0, a0, a1
80001004: 0c057557     	vsetvli	a0, a0, e8, m1, ta, ma
80001008: fca43023     	sd	a0, -0x40(s0)
8000100c: fe043503     	ld	a0, -0x20(s0)
80001010: fcc42583     	lw	a1, -0x34(s0)
80001014: 952e         	add	a0, a0, a1
80001016: fc043583     	ld	a1, -0x40(s0)
8000101a: 0805f057     	vsetvli	zero, a1, e8, m1, tu, ma
8000101e: 02050407     	vle8.v	v8, (a0)
80001022: c2202573     	csrr	a0, vlenb
80001026: 40a40533     	sub	a0, s0, a0
8000102a: fc050513     	addi	a0, a0, -0x40
8000102e: 0c0075d7     	vsetvli	a1, zero, e8, m1, ta, ma
80001032: 02050427     	vse8.v	v8, (a0)
80001036: 02050507     	vle8.v	v10, (a0)
8000103a: fc043583     	ld	a1, -0x40(s0)
8000103e: 4501         	li	a0, 0x0
80001040: 0805f057     	vsetvli	zero, a1, e8, m1, tu, ma
80001044: c6a56457     	vwadd.vx	v8, v10, a0
80001048: c22025f3     	csrr	a1, vlenb
8000104c: 00159613     	slli	a2, a1, 0x1
80001050: 95b2         	add	a1, a1, a2
80001052: 40b405b3     	sub	a1, s0, a1
80001056: fc058593     	addi	a1, a1, -0x40
8000105a: 0c907657     	vsetvli	a2, zero, e16, m2, ta, ma
8000105e: 0205d427     	vse16.v	v8, (a1)
80001062: 0205d607     	vle16.v	v12, (a1)
80001066: fc043583     	ld	a1, -0x40(s0)
8000106a: 0895f057     	vsetvli	zero, a1, e16, m2, tu, ma
8000106e: c6c56457     	vwadd.vx	v8, v12, a0
80001072: c2202573     	csrr	a0, vlenb
80001076: 00351593     	slli	a1, a0, 0x3
8000107a: 40a58533     	sub	a0, a1, a0
8000107e: 40a40533     	sub	a0, s0, a0
80001082: fc050513     	addi	a0, a0, -0x40
80001086: 0d2075d7     	vsetvli	a1, zero, e32, m4, ta, ma
8000108a: 02056427     	vse32.v	v8, (a0)
8000108e: 02056607     	vle32.v	v12, (a0)
80001092: fd042503     	lw	a0, -0x30(s0)
80001096: fc043583     	ld	a1, -0x40(s0)
8000109a: 0925f057     	vsetvli	zero, a1, e32, m4, tu, ma
8000109e: 96c56457     	vmul.vx	v8, v12, a0
800010a2: c22025f3     	csrr	a1, vlenb
800010a6: 452d         	li	a0, 0xb
800010a8: 02a585b3     	mul	a1, a1, a0
800010ac: 40b405b3     	sub	a1, s0, a1
800010b0: fc058593     	addi	a1, a1, -0x40
800010b4: 0d207557     	vsetvli	a0, zero, e32, m4, ta, ma
800010b8: 0205e427     	vse32.v	v8, (a1)
800010bc: fd843503     	ld	a0, -0x28(s0)
800010c0: fcc42603     	lw	a2, -0x34(s0)
800010c4: 060a         	slli	a2, a2, 0x2
800010c6: 9532         	add	a0, a0, a2
800010c8: 0205e407     	vle32.v	v8, (a1)
800010cc: fc043583     	ld	a1, -0x40(s0)
800010d0: 0d25f057     	vsetvli	zero, a1, e32, m4, ta, ma
800010d4: 02056427     	vse32.v	v8, (a0)
800010d8: fc043583     	ld	a1, -0x40(s0)
800010dc: fcc42503     	lw	a0, -0x34(s0)
800010e0: 9d2d         	addw	a0, a0, a1
800010e2: fca42623     	sw	a0, -0x34(s0)
800010e6: b719         	j	0x80000fec <mf_dequantize_i8_to_i32+0x2e>
800010e8: fc040113     	addi	sp, s0, -0x40
800010ec: 70e2         	ld	ra, 0x38(sp)
800010ee: 7442         	ld	s0, 0x30(sp)
800010f0: 6121         	addi	sp, sp, 0x40
800010f2: 8082         	ret

00000000800010f4 <mf_bias_add_i32>:
; mf_bias_add_i32():
800010f4: 715d         	addi	sp, sp, -0x50
800010f6: e486         	sd	ra, 0x48(sp)
800010f8: e0a2         	sd	s0, 0x40(sp)
800010fa: 0880         	addi	s0, sp, 0x50
800010fc: c2202773     	csrr	a4, vlenb
80001100: 070e         	slli	a4, a4, 0x3
80001102: 40e10133     	sub	sp, sp, a4
80001106: fea43023     	sd	a0, -0x20(s0)
8000110a: fcb43c23     	sd	a1, -0x28(s0)
8000110e: fcc42a23     	sw	a2, -0x2c(s0)
80001112: fcd42823     	sw	a3, -0x30(s0)
80001116: 4501         	li	a0, 0x0
80001118: fca42623     	sw	a0, -0x34(s0)
8000111c: a009         	j	0x8000111e <mf_bias_add_i32+0x2a>
8000111e: fcc42503     	lw	a0, -0x34(s0)
80001122: fd442583     	lw	a1, -0x2c(s0)
80001126: 0eb55563     	bge	a0, a1, 0x80001210 <mf_bias_add_i32+0x11c>
8000112a: a009         	j	0x8000112c <mf_bias_add_i32+0x38>
8000112c: fd843503     	ld	a0, -0x28(s0)
80001130: fcc42583     	lw	a1, -0x34(s0)
80001134: 058a         	slli	a1, a1, 0x2
80001136: 952e         	add	a0, a0, a1
80001138: 4108         	lw	a0, 0x0(a0)
8000113a: fca42423     	sw	a0, -0x38(s0)
8000113e: fe043503     	ld	a0, -0x20(s0)
80001142: fcc42583     	lw	a1, -0x34(s0)
80001146: fd042603     	lw	a2, -0x30(s0)
8000114a: 02c585b3     	mul	a1, a1, a2
8000114e: 058a         	slli	a1, a1, 0x2
80001150: 952e         	add	a0, a0, a1
80001152: fca43023     	sd	a0, -0x40(s0)
80001156: 4501         	li	a0, 0x0
80001158: faa42e23     	sw	a0, -0x44(s0)
8000115c: a009         	j	0x8000115e <mf_bias_add_i32+0x6a>
8000115e: fbc42503     	lw	a0, -0x44(s0)
80001162: fd042583     	lw	a1, -0x30(s0)
80001166: 08b55e63     	bge	a0, a1, 0x80001202 <mf_bias_add_i32+0x10e>
8000116a: a009         	j	0x8000116c <mf_bias_add_i32+0x78>
8000116c: fd042503     	lw	a0, -0x30(s0)
80001170: fbc42583     	lw	a1, -0x44(s0)
80001174: 9d0d         	subw	a0, a0, a1
80001176: 0d257557     	vsetvli	a0, a0, e32, m4, ta, ma
8000117a: faa43823     	sd	a0, -0x50(s0)
8000117e: fc043503     	ld	a0, -0x40(s0)
80001182: fbc42583     	lw	a1, -0x44(s0)
80001186: 058a         	slli	a1, a1, 0x2
80001188: 952e         	add	a0, a0, a1
8000118a: fb043583     	ld	a1, -0x50(s0)
8000118e: 0925f057     	vsetvli	zero, a1, e32, m4, tu, ma
80001192: 02056407     	vle32.v	v8, (a0)
80001196: c2202573     	csrr	a0, vlenb
8000119a: 050a         	slli	a0, a0, 0x2
8000119c: 40a40533     	sub	a0, s0, a0
800011a0: fb050513     	addi	a0, a0, -0x50
800011a4: 0d2075d7     	vsetvli	a1, zero, e32, m4, ta, ma
800011a8: 02056427     	vse32.v	v8, (a0)
800011ac: 02056607     	vle32.v	v12, (a0)
800011b0: fc842503     	lw	a0, -0x38(s0)
800011b4: fb043583     	ld	a1, -0x50(s0)
800011b8: 0925f057     	vsetvli	zero, a1, e32, m4, tu, ma
800011bc: 02c54457     	vadd.vx	v8, v12, a0
800011c0: c22025f3     	csrr	a1, vlenb
800011c4: 058e         	slli	a1, a1, 0x3
800011c6: 40b405b3     	sub	a1, s0, a1
800011ca: fb058593     	addi	a1, a1, -0x50
800011ce: 0d207557     	vsetvli	a0, zero, e32, m4, ta, ma
800011d2: 0205e427     	vse32.v	v8, (a1)
800011d6: fc043503     	ld	a0, -0x40(s0)
800011da: fbc42603     	lw	a2, -0x44(s0)
800011de: 060a         	slli	a2, a2, 0x2
800011e0: 9532         	add	a0, a0, a2
800011e2: 0205e407     	vle32.v	v8, (a1)
800011e6: fb043583     	ld	a1, -0x50(s0)
800011ea: 0d25f057     	vsetvli	zero, a1, e32, m4, ta, ma
800011ee: 02056427     	vse32.v	v8, (a0)
800011f2: fb043583     	ld	a1, -0x50(s0)
800011f6: fbc42503     	lw	a0, -0x44(s0)
800011fa: 9d2d         	addw	a0, a0, a1
800011fc: faa42e23     	sw	a0, -0x44(s0)
80001200: bfb9         	j	0x8000115e <mf_bias_add_i32+0x6a>
80001202: a009         	j	0x80001204 <mf_bias_add_i32+0x110>
80001204: fcc42503     	lw	a0, -0x34(s0)
80001208: 2505         	addiw	a0, a0, 0x1
8000120a: fca42623     	sw	a0, -0x34(s0)
8000120e: bf01         	j	0x8000111e <mf_bias_add_i32+0x2a>
80001210: fb040113     	addi	sp, s0, -0x50
80001214: 60a6         	ld	ra, 0x48(sp)
80001216: 6406         	ld	s0, 0x40(sp)
80001218: 6161         	addi	sp, sp, 0x50
8000121a: 8082         	ret
