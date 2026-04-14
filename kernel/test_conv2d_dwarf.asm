; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:60
;     switch (acc_idx)
80002fc6: 02b50863     	beq	a0, a1, 0x80002ff6 <mf_tmma_ii8+0x74>
80002fca: a82d         	j	0x80003004 <mf_tmma_ii8+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:63
;         __builtin_riscv_mf_tmma_ii8_i(0, (void *)A, (void *)B);
80002fcc: fe043503     	ld	a0, -0x20(s0)
80002fd0: fd843583     	ld	a1, -0x28(s0)
80002fd4: 12b5400b     	tmma.ii8.i	0x0, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:64
;         break;
80002fd8: a035         	j	0x80003004 <mf_tmma_ii8+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:66
;         __builtin_riscv_mf_tmma_ii8_i(1, (void *)A, (void *)B);
80002fda: fe043503     	ld	a0, -0x20(s0)
80002fde: fd843583     	ld	a1, -0x28(s0)
80002fe2: 12b5408b     	tmma.ii8.i	0x1, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:67
;         break;
80002fe6: a839         	j	0x80003004 <mf_tmma_ii8+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:69
;         __builtin_riscv_mf_tmma_ii8_i(2, (void *)A, (void *)B);
80002fe8: fe043503     	ld	a0, -0x20(s0)
80002fec: fd843583     	ld	a1, -0x28(s0)
80002ff0: 12b5410b     	tmma.ii8.i	0x2, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:70
;         break;
80002ff4: a801         	j	0x80003004 <mf_tmma_ii8+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:72
;         __builtin_riscv_mf_tmma_ii8_i(3, (void *)A, (void *)B);
80002ff6: fe043503     	ld	a0, -0x20(s0)
80002ffa: fd843583     	ld	a1, -0x28(s0)
80002ffe: 12b5418b     	tmma.ii8.i	0x3, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:73
;         break;
80003002: a009         	j	0x80003004 <mf_tmma_ii8+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:75
; }
80003004: fd040113     	addi	sp, s0, -0x30
80003008: 70a2         	ld	ra, 0x28(sp)
8000300a: 7402         	ld	s0, 0x20(sp)
8000300c: 6145         	addi	sp, sp, 0x30
8000300e: 8082         	ret

0000000080003010 <mf_tstore32>:
; mf_tstore32():
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:253
; {
80003010: 7179         	addi	sp, sp, -0x30
80003012: f406         	sd	ra, 0x28(sp)
80003014: f022         	sd	s0, 0x20(sp)
80003016: 1800         	addi	s0, sp, 0x30
80003018: fea42623     	sw	a0, -0x14(s0)
8000301c: feb43023     	sd	a1, -0x20(s0)
80003020: fcc43c23     	sd	a2, -0x28(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:254
;     switch (acc_idx)
80003024: fec42503     	lw	a0, -0x14(s0)
80003028: fca43823     	sd	a0, -0x30(s0)
8000302c: c51d         	beqz	a0, 0x8000305a <mf_tstore32+0x4a>
8000302e: a009         	j	0x80003030 <mf_tstore32+0x20>
80003030: fd043503     	ld	a0, -0x30(s0)
80003034: 2501         	sext.w	a0, a0
80003036: 4585         	li	a1, 0x1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:254
;     switch (acc_idx)
80003038: 02b50863     	beq	a0, a1, 0x80003068 <mf_tstore32+0x58>
8000303c: a009         	j	0x8000303e <mf_tstore32+0x2e>
8000303e: fd043503     	ld	a0, -0x30(s0)
80003042: 2501         	sext.w	a0, a0
80003044: 4589         	li	a1, 0x2
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:254
;     switch (acc_idx)
80003046: 02b50863     	beq	a0, a1, 0x80003076 <mf_tstore32+0x66>
8000304a: a009         	j	0x8000304c <mf_tstore32+0x3c>
8000304c: fd043503     	ld	a0, -0x30(s0)
80003050: 2501         	sext.w	a0, a0
80003052: 458d         	li	a1, 0x3
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:254
;     switch (acc_idx)
80003054: 02b50863     	beq	a0, a1, 0x80003084 <mf_tstore32+0x74>
80003058: a82d         	j	0x80003092 <mf_tstore32+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:257
;         __builtin_riscv_mf_tstore32_ix(0, dst, (unsigned long)bytes);
8000305a: fe043503     	ld	a0, -0x20(s0)
8000305e: fd843583     	ld	a1, -0x28(s0)
80003062: 86b5100b     	tstore32.ix	0x0, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:258
;         break;
80003066: a035         	j	0x80003092 <mf_tstore32+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:260
;         __builtin_riscv_mf_tstore32_ix(1, dst, (unsigned long)bytes);
80003068: fe043503     	ld	a0, -0x20(s0)
8000306c: fd843583     	ld	a1, -0x28(s0)
80003070: 86b5108b     	tstore32.ix	0x1, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:261
;         break;
80003074: a839         	j	0x80003092 <mf_tstore32+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:263
;         __builtin_riscv_mf_tstore32_ix(2, dst, (unsigned long)bytes);
80003076: fe043503     	ld	a0, -0x20(s0)
8000307a: fd843583     	ld	a1, -0x28(s0)
8000307e: 86b5110b     	tstore32.ix	0x2, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:264
;         break;
80003082: a801         	j	0x80003092 <mf_tstore32+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:266
;         __builtin_riscv_mf_tstore32_ix(3, dst, (unsigned long)bytes);
80003084: fe043503     	ld	a0, -0x20(s0)
80003088: fd843583     	ld	a1, -0x28(s0)
8000308c: 86b5118b     	tstore32.ix	0x3, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:267
;         break;
80003090: a009         	j	0x80003092 <mf_tstore32+0x82>
; /home/heyi/heyi/matrixflow/kernel/./mf_gemm.h:269
; }
80003092: fd040113     	addi	sp, s0, -0x30
80003096: 70a2         	ld	ra, 0x28(sp)
80003098: 7402         	ld	s0, 0x20(sp)
8000309a: 6145         	addi	sp, sp, 0x30
8000309c: 8082         	ret

000000008000309e <mf_conv2d_3x3_i8>:
; mf_conv2d_3x3_i8():
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:162
;                                      mf_l1_buf_t workspace) {
8000309e: 7125         	addi	sp, sp, -0x1a0
800030a0: ef06         	sd	ra, 0x198(sp)
800030a2: eb22         	sd	s0, 0x190(sp)
800030a4: 1300         	addi	s0, sp, 0x1a0
800030a6: 00843303     	ld	t1, 0x8(s0)
800030aa: 00043283     	ld	t0, 0x0(s0)
800030ae: fe643023     	sd	t1, -0x20(s0)
800030b2: fc543c23     	sd	t0, -0x28(s0)
800030b6: fca43823     	sd	a0, -0x30(s0)
800030ba: fcb43423     	sd	a1, -0x38(s0)
800030be: fcc43023     	sd	a2, -0x40(s0)
800030c2: fad42e23     	sw	a3, -0x44(s0)
800030c6: fae42c23     	sw	a4, -0x48(s0)
800030ca: faf42a23     	sw	a5, -0x4c(s0)
800030ce: fb042823     	sw	a6, -0x50(s0)
800030d2: fb142623     	sw	a7, -0x54(s0)
800030d6: 450d         	li	a0, 0x3
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:163
;     const int kH = 3, kW = 3;
800030d8: faa42423     	sw	a0, -0x58(s0)
800030dc: faa42223     	sw	a0, -0x5c(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:164
;     const int outH = H + 2 * pad - kH + 1;
800030e0: fb442503     	lw	a0, -0x4c(s0)
800030e4: fac42583     	lw	a1, -0x54(s0)
800030e8: 0015959b     	slliw	a1, a1, 0x1
800030ec: 9d2d         	addw	a0, a0, a1
800030ee: 3579         	addiw	a0, a0, -0x2
800030f0: faa42023     	sw	a0, -0x60(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:165
;     const int outW = W + 2 * pad - kW + 1;
800030f4: fb042503     	lw	a0, -0x50(s0)
800030f8: fac42583     	lw	a1, -0x54(s0)
800030fc: 0015959b     	slliw	a1, a1, 0x1
80003100: 9d2d         	addw	a0, a0, a1
80003102: 3579         	addiw	a0, a0, -0x2
80003104: f8a42e23     	sw	a0, -0x64(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:166
;     const int K = C_in * kH * kW;
80003108: fbc42583     	lw	a1, -0x44(s0)
8000310c: 0015951b     	slliw	a0, a1, 0x1
80003110: 9da9         	addw	a1, a1, a0
80003112: 0015951b     	slliw	a0, a1, 0x1
80003116: 9d2d         	addw	a0, a0, a1
80003118: f8a42c23     	sw	a0, -0x68(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:167
;     const int K_padded = MF_DIV_CEIL(K, MF_TILE_K) * MF_TILE_K;
8000311c: f9842503     	lw	a0, -0x68(s0)
80003120: 257d         	addiw	a0, a0, 0x1f
80003122: 41f5559b     	sraiw	a1, a0, 0x1f
80003126: 01b5d59b     	srliw	a1, a1, 0x1b
8000312a: 9d2d         	addw	a0, a0, a1
8000312c: 9901         	andi	a0, a0, -0x20
8000312e: f8a42a23     	sw	a0, -0x6c(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:168
;     const int N = outH * outW;
80003132: fa042503     	lw	a0, -0x60(s0)
80003136: f9c42583     	lw	a1, -0x64(s0)
8000313a: 02b5053b     	mulw	a0, a0, a1
8000313e: f8a42823     	sw	a0, -0x70(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:170
;     if (outH <= 0 || outW <= 0)
80003142: fa042583     	lw	a1, -0x60(s0)
80003146: 4501         	li	a0, 0x0
80003148: 00b55963     	bge	a0, a1, 0x8000315a <mf_conv2d_3x3_i8+0xbc>
8000314c: a009         	j	0x8000314e <mf_conv2d_3x3_i8+0xb0>
8000314e: f9c42583     	lw	a1, -0x64(s0)
80003152: 4501         	li	a0, 0x0
80003154: 00b54763     	blt	a0, a1, 0x80003162 <mf_conv2d_3x3_i8+0xc4>
80003158: a009         	j	0x8000315a <mf_conv2d_3x3_i8+0xbc>
8000315a: 4509         	li	a0, 0x2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:171
;         return MF_ERR_INVALID_PARAM;
8000315c: fea42623     	sw	a0, -0x14(s0)
80003160: ad09         	j	0x80003772 <mf_conv2d_3x3_i8+0x6d4>
80003162: 4541         	li	a0, 0x10
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:173
;     const int tile_m = MF_TILE_M;
80003164: f8a42623     	sw	a0, -0x74(s0)
80003168: 04000513     	li	a0, 0x40
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:174
;     const int tile_n = MF_TILE_N;
8000316c: f8a42423     	sw	a0, -0x78(s0)
80003170: 02000513     	li	a0, 0x20
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:175
;     const int tile_k = MF_TILE_K;
80003174: f8a42223     	sw	a0, -0x7c(s0)
80003178: 4511         	li	a0, 0x4
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:176
;     const int c_elem = 4;          /* int32 accumulator */
8000317a: f8a42023     	sw	a0, -0x80(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:177
;     const int padH = H + 2 * pad;
8000317e: fb442503     	lw	a0, -0x4c(s0)
80003182: fac42583     	lw	a1, -0x54(s0)
80003186: 0015959b     	slliw	a1, a1, 0x1
8000318a: 9d2d         	addw	a0, a0, a1
8000318c: f6a42e23     	sw	a0, -0x84(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:178
;     const int padW = W + 2 * pad;
80003190: fb042503     	lw	a0, -0x50(s0)
80003194: fac42583     	lw	a1, -0x54(s0)
80003198: 0015959b     	slliw	a1, a1, 0x1
8000319c: 9d2d         	addw	a0, a0, a1
8000319e: f6a42c23     	sw	a0, -0x88(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:181
;     mf_l1_alloc_init(&alloc, workspace);
800031a2: fe043603     	ld	a2, -0x20(s0)
800031a6: fd843583     	ld	a1, -0x28(s0)
800031aa: f6040513     	addi	a0, s0, -0xa0
800031ae: eaa43023     	sd	a0, -0x160(s0)
800031b2: 5d0000ef     	jal	0x80003782 <mf_l1_alloc_init>
800031b6: ea043503     	ld	a0, -0x160(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:184
;     size_t pad_bytes = (size_t)C_in * padH * padW;
800031ba: fbc42583     	lw	a1, -0x44(s0)
800031be: f7c42603     	lw	a2, -0x84(s0)
800031c2: 02c585b3     	mul	a1, a1, a2
800031c6: f7842603     	lw	a2, -0x88(s0)
800031ca: 02c585b3     	mul	a1, a1, a2
800031ce: f4b43c23     	sd	a1, -0xa8(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:185
;     int8_t *padded = (int8_t *)mf_l1_alloc(&alloc, pad_bytes);
800031d2: f5843583     	ld	a1, -0xa8(s0)
800031d6: 5e8000ef     	jal	0x800037be <mf_l1_alloc>
800031da: f4a43823     	sd	a0, -0xb0(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:186
;     if (!padded) return MF_ERR_L1_TOO_SMALL;
800031de: f5043503     	ld	a0, -0xb0(s0)
800031e2: e511         	bnez	a0, 0x800031ee <mf_conv2d_3x3_i8+0x150>
800031e4: a009         	j	0x800031e6 <mf_conv2d_3x3_i8+0x148>
800031e6: 4505         	li	a0, 0x1
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:186
;     if (!padded) return MF_ERR_L1_TOO_SMALL;
800031e8: fea42623     	sw	a0, -0x14(s0)
800031ec: a359         	j	0x80003772 <mf_conv2d_3x3_i8+0x6d4>
800031ee: f6040513     	addi	a0, s0, -0xa0
800031f2: e8a43c23     	sd	a0, -0x168(s0)
800031f6: 20000593     	li	a1, 0x200
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:189
;     uint8_t *a_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_A_BYTES);
800031fa: 5c4000ef     	jal	0x800037be <mf_l1_alloc>
800031fe: 85aa         	mv	a1, a0
80003200: e9843503     	ld	a0, -0x168(s0)
80003204: f4b43423     	sd	a1, -0xb8(s0)
80003208: 4585         	li	a1, 0x1
8000320a: 05ae         	slli	a1, a1, 0xb
8000320c: e8b43823     	sd	a1, -0x170(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:190
;     uint8_t *b_tmp = (uint8_t *)mf_l1_alloc(&alloc, tile_k * tile_n);
80003210: 5ae000ef     	jal	0x800037be <mf_l1_alloc>
80003214: e9043583     	ld	a1, -0x170(s0)
80003218: 862a         	mv	a2, a0
8000321a: e9843503     	ld	a0, -0x168(s0)
8000321e: f4c43023     	sd	a2, -0xc0(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:191
;     uint8_t *b_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_B_BYTES);
80003222: 59c000ef     	jal	0x800037be <mf_l1_alloc>
80003226: 85aa         	mv	a1, a0
80003228: e9843503     	ld	a0, -0x168(s0)
8000322c: f2b43c23     	sd	a1, -0xc8(s0)
80003230: 6585         	lui	a1, 0x1
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:192
;     uint8_t *c_tmp = (uint8_t *)mf_l1_alloc(&alloc, MF_ACC_OUT_I8);
80003232: 58c000ef     	jal	0x800037be <mf_l1_alloc>
80003236: f2a43823     	sd	a0, -0xd0(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:193
;     if (!a_pad || !b_tmp || !b_pad || !c_tmp)
8000323a: f4843503     	ld	a0, -0xb8(s0)
8000323e: cd11         	beqz	a0, 0x8000325a <mf_conv2d_3x3_i8+0x1bc>
80003240: a009         	j	0x80003242 <mf_conv2d_3x3_i8+0x1a4>
80003242: f4043503     	ld	a0, -0xc0(s0)
80003246: c911         	beqz	a0, 0x8000325a <mf_conv2d_3x3_i8+0x1bc>
80003248: a009         	j	0x8000324a <mf_conv2d_3x3_i8+0x1ac>
8000324a: f3843503     	ld	a0, -0xc8(s0)
8000324e: c511         	beqz	a0, 0x8000325a <mf_conv2d_3x3_i8+0x1bc>
80003250: a009         	j	0x80003252 <mf_conv2d_3x3_i8+0x1b4>
80003252: f3043503     	ld	a0, -0xd0(s0)
80003256: e511         	bnez	a0, 0x80003262 <mf_conv2d_3x3_i8+0x1c4>
80003258: a009         	j	0x8000325a <mf_conv2d_3x3_i8+0x1bc>
8000325a: 4505         	li	a0, 0x1
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:194
;         return MF_ERR_L1_TOO_SMALL;
8000325c: fea42623     	sw	a0, -0x14(s0)
80003260: ab09         	j	0x80003772 <mf_conv2d_3x3_i8+0x6d4>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:197
;     mf_memset(padded, 0, pad_bytes);
80003262: f5043503     	ld	a0, -0xb0(s0)
80003266: f5843603     	ld	a2, -0xa8(s0)
8000326a: 4581         	li	a1, 0x0
8000326c: e8b43423     	sd	a1, -0x178(s0)
80003270: dc2ff0ef     	jal	0x80002832 <mf_memset>
80003274: e8843503     	ld	a0, -0x178(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:198
;     for (int c = 0; c < C_in; c++)
80003278: f2a42623     	sw	a0, -0xd4(s0)
8000327c: a009         	j	0x8000327e <mf_conv2d_3x3_i8+0x1e0>
8000327e: f2c42503     	lw	a0, -0xd4(s0)
80003282: fbc42583     	lw	a1, -0x44(s0)
80003286: 08b55363     	bge	a0, a1, 0x8000330c <mf_conv2d_3x3_i8+0x26e>
8000328a: a009         	j	0x8000328c <mf_conv2d_3x3_i8+0x1ee>
8000328c: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:199
;         for (int h = 0; h < H; h++)
8000328e: f2a42423     	sw	a0, -0xd8(s0)
80003292: a009         	j	0x80003294 <mf_conv2d_3x3_i8+0x1f6>
80003294: f2842503     	lw	a0, -0xd8(s0)
80003298: fb442583     	lw	a1, -0x4c(s0)
8000329c: 06b55163     	bge	a0, a1, 0x800032fe <mf_conv2d_3x3_i8+0x260>
800032a0: a009         	j	0x800032a2 <mf_conv2d_3x3_i8+0x204>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:201
;                 padded + c * padH * padW + (h + pad) * padW + pad,
800032a2: f5043503     	ld	a0, -0xb0(s0)
800032a6: f2c42603     	lw	a2, -0xd4(s0)
800032aa: f7c42583     	lw	a1, -0x84(s0)
800032ae: 02b605bb     	mulw	a1, a2, a1
800032b2: f7842783     	lw	a5, -0x88(s0)
800032b6: 02f585bb     	mulw	a1, a1, a5
800032ba: 952e         	add	a0, a0, a1
800032bc: f2842683     	lw	a3, -0xd8(s0)
800032c0: fac42583     	lw	a1, -0x54(s0)
800032c4: 00b6873b     	addw	a4, a3, a1
800032c8: 02f7073b     	mulw	a4, a4, a5
800032cc: 953a         	add	a0, a0, a4
800032ce: 952e         	add	a0, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:202
;                 (void *)(input + c * H * W + h * W),
800032d0: fd043583     	ld	a1, -0x30(s0)
800032d4: fb442703     	lw	a4, -0x4c(s0)
800032d8: 02e6073b     	mulw	a4, a2, a4
800032dc: fb042603     	lw	a2, -0x50(s0)
800032e0: 02c7073b     	mulw	a4, a4, a2
800032e4: 95ba         	add	a1, a1, a4
800032e6: 02c686bb     	mulw	a3, a3, a2
800032ea: 95b6         	add	a1, a1, a3
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:200
;             __builtin_riscv_mf_dma_x(
800032ec: 00c5d50b     	dma.x	a0, a1, a2
800032f0: a009         	j	0x800032f2 <mf_conv2d_3x3_i8+0x254>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:199
;         for (int h = 0; h < H; h++)
800032f2: f2842503     	lw	a0, -0xd8(s0)
800032f6: 2505         	addiw	a0, a0, 0x1
800032f8: f2a42423     	sw	a0, -0xd8(s0)
800032fc: bf61         	j	0x80003294 <mf_conv2d_3x3_i8+0x1f6>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:203
;                 (unsigned long)W);
800032fe: a009         	j	0x80003300 <mf_conv2d_3x3_i8+0x262>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:198
;     for (int c = 0; c < C_in; c++)
80003300: f2c42503     	lw	a0, -0xd4(s0)
80003304: 2505         	addiw	a0, a0, 0x1
80003306: f2a42623     	sw	a0, -0xd4(s0)
8000330a: bf95         	j	0x8000327e <mf_conv2d_3x3_i8+0x1e0>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:204
;     __builtin_riscv_mf_dma_sync();
8000330c: 8000000b     	dma.sync
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:207
;     int m_tiles = MF_DIV_CEIL(C_out, tile_m);
80003310: fb842503     	lw	a0, -0x48(s0)
80003314: 253d         	addiw	a0, a0, 0xf
80003316: 41f5559b     	sraiw	a1, a0, 0x1f
8000331a: 01c5d59b     	srliw	a1, a1, 0x1c
8000331e: 9d2d         	addw	a0, a0, a1
80003320: 4045551b     	sraiw	a0, a0, 0x4
80003324: f2a42223     	sw	a0, -0xdc(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:208
;     int n_tiles = MF_DIV_CEIL(N, tile_n);
80003328: f9042503     	lw	a0, -0x70(s0)
8000332c: 03f5051b     	addiw	a0, a0, 0x3f
80003330: 41f5559b     	sraiw	a1, a0, 0x1f
80003334: 01a5d59b     	srliw	a1, a1, 0x1a
80003338: 9d2d         	addw	a0, a0, a1
8000333a: 4065551b     	sraiw	a0, a0, 0x6
8000333e: f2a42023     	sw	a0, -0xe0(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:209
;     int k_tiles = K_padded / tile_k;
80003342: f9442503     	lw	a0, -0x6c(s0)
80003346: 00151593     	slli	a1, a0, 0x1
8000334a: 91ed         	srli	a1, a1, 0x3b
8000334c: 9d2d         	addw	a0, a0, a1
8000334e: 4055551b     	sraiw	a0, a0, 0x5
80003352: f0a42e23     	sw	a0, -0xe4(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:211
;     const uint8_t *w_base = (const uint8_t *)weight;
80003356: fc843503     	ld	a0, -0x38(s0)
8000335a: f0a43823     	sd	a0, -0xf0(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:212
;     uint8_t *c_base = (uint8_t *)output;
8000335e: fc043503     	ld	a0, -0x40(s0)
80003362: f0a43423     	sd	a0, -0xf8(s0)
80003366: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:215
;     for (int mg = 0; mg < m_tiles; mg += MF_NUM_ACC) {
80003368: f0a42223     	sw	a0, -0xfc(s0)
8000336c: a009         	j	0x8000336e <mf_conv2d_3x3_i8+0x2d0>
8000336e: f0442503     	lw	a0, -0xfc(s0)
80003372: f2442583     	lw	a1, -0xdc(s0)
80003376: 3eb55a63     	bge	a0, a1, 0x8000376a <mf_conv2d_3x3_i8+0x6cc>
8000337a: a009         	j	0x8000337c <mf_conv2d_3x3_i8+0x2de>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:216
;         int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);
8000337c: f2442503     	lw	a0, -0xdc(s0)
80003380: f0442583     	lw	a1, -0xfc(s0)
80003384: 9d0d         	subw	a0, a0, a1
80003386: 4595         	li	a1, 0x5
80003388: 00b54763     	blt	a0, a1, 0x80003396 <mf_conv2d_3x3_i8+0x2f8>
8000338c: a009         	j	0x8000338e <mf_conv2d_3x3_i8+0x2f0>
8000338e: 4511         	li	a0, 0x4
80003390: e8a43023     	sd	a0, -0x180(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:216
;         int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);
80003394: a809         	j	0x800033a6 <mf_conv2d_3x3_i8+0x308>
80003396: f2442503     	lw	a0, -0xdc(s0)
8000339a: f0442583     	lw	a1, -0xfc(s0)
8000339e: 9d0d         	subw	a0, a0, a1
800033a0: e8a43023     	sd	a0, -0x180(s0)
800033a4: a009         	j	0x800033a6 <mf_conv2d_3x3_i8+0x308>
800033a6: e8043503     	ld	a0, -0x180(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:216
;         int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);
800033aa: f0a42023     	sw	a0, -0x100(s0)
800033ae: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:218
;         for (int nt = 0; nt < n_tiles; nt++) {
800033b0: eea42e23     	sw	a0, -0x104(s0)
800033b4: a009         	j	0x800033b6 <mf_conv2d_3x3_i8+0x318>
800033b6: efc42503     	lw	a0, -0x104(s0)
800033ba: f2042583     	lw	a1, -0xe0(s0)
800033be: 38b55f63     	bge	a0, a1, 0x8000375c <mf_conv2d_3x3_i8+0x6be>
800033c2: a009         	j	0x800033c4 <mf_conv2d_3x3_i8+0x326>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:219
;             int n_start = nt * tile_n;
800033c4: efc42503     	lw	a0, -0x104(s0)
800033c8: 0065151b     	slliw	a0, a0, 0x6
800033cc: eea42c23     	sw	a0, -0x108(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:220
;             int actual_n = MF_MIN(tile_n, N - n_start);
800033d0: f9042503     	lw	a0, -0x70(s0)
800033d4: ef842583     	lw	a1, -0x108(s0)
800033d8: 9d0d         	subw	a0, a0, a1
800033da: 04100593     	li	a1, 0x41
800033de: 00b54863     	blt	a0, a1, 0x800033ee <mf_conv2d_3x3_i8+0x350>
800033e2: a009         	j	0x800033e4 <mf_conv2d_3x3_i8+0x346>
800033e4: 04000513     	li	a0, 0x40
800033e8: e6a43c23     	sd	a0, -0x188(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:220
;             int actual_n = MF_MIN(tile_n, N - n_start);
800033ec: a809         	j	0x800033fe <mf_conv2d_3x3_i8+0x360>
800033ee: f9042503     	lw	a0, -0x70(s0)
800033f2: ef842583     	lw	a1, -0x108(s0)
800033f6: 9d0d         	subw	a0, a0, a1
800033f8: e6a43c23     	sd	a0, -0x188(s0)
800033fc: a009         	j	0x800033fe <mf_conv2d_3x3_i8+0x360>
800033fe: e7843503     	ld	a0, -0x188(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:220
;             int actual_n = MF_MIN(tile_n, N - n_start);
80003402: eea42a23     	sw	a0, -0x10c(s0)
80003406: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:222
;             for (int a = 0; a < m_batch; a++)
80003408: eea42823     	sw	a0, -0x110(s0)
8000340c: a009         	j	0x8000340e <mf_conv2d_3x3_i8+0x370>
8000340e: ef042503     	lw	a0, -0x110(s0)
80003412: f0042583     	lw	a1, -0x100(s0)
80003416: 00b55e63     	bge	a0, a1, 0x80003432 <mf_conv2d_3x3_i8+0x394>
8000341a: a009         	j	0x8000341c <mf_conv2d_3x3_i8+0x37e>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:223
;                 mf_tzero(a);
8000341c: ef042503     	lw	a0, -0x110(s0)
80003420: a2bff0ef     	jal	0x80002e4a <mf_tzero>
80003424: a009         	j	0x80003426 <mf_conv2d_3x3_i8+0x388>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:222
;             for (int a = 0; a < m_batch; a++)
80003426: ef042503     	lw	a0, -0x110(s0)
8000342a: 2505         	addiw	a0, a0, 0x1
8000342c: eea42823     	sw	a0, -0x110(s0)
80003430: bff9         	j	0x8000340e <mf_conv2d_3x3_i8+0x370>
80003432: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:225
;             for (int kt = 0; kt < k_tiles; kt++) {
80003434: eea42623     	sw	a0, -0x114(s0)
80003438: a009         	j	0x8000343a <mf_conv2d_3x3_i8+0x39c>
8000343a: eec42503     	lw	a0, -0x114(s0)
8000343e: f1c42583     	lw	a1, -0xe4(s0)
80003442: 22b55163     	bge	a0, a1, 0x80003664 <mf_conv2d_3x3_i8+0x5c6>
80003446: a009         	j	0x80003448 <mf_conv2d_3x3_i8+0x3aa>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:226
;                 int k_start = kt * tile_k;
80003448: eec42503     	lw	a0, -0x114(s0)
8000344c: 0055151b     	slliw	a0, a0, 0x5
80003450: eea42423     	sw	a0, -0x118(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:227
;                 int k_valid = MF_MIN(tile_k, K - k_start);
80003454: f9842503     	lw	a0, -0x68(s0)
80003458: ee842583     	lw	a1, -0x118(s0)
8000345c: 9d0d         	subw	a0, a0, a1
8000345e: 02100593     	li	a1, 0x21
80003462: 00b54863     	blt	a0, a1, 0x80003472 <mf_conv2d_3x3_i8+0x3d4>
80003466: a009         	j	0x80003468 <mf_conv2d_3x3_i8+0x3ca>
80003468: 02000513     	li	a0, 0x20
8000346c: e6a43823     	sd	a0, -0x190(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:227
;                 int k_valid = MF_MIN(tile_k, K - k_start);
80003470: a809         	j	0x80003482 <mf_conv2d_3x3_i8+0x3e4>
80003472: f9842503     	lw	a0, -0x68(s0)
80003476: ee842583     	lw	a1, -0x118(s0)
8000347a: 9d0d         	subw	a0, a0, a1
8000347c: e6a43823     	sd	a0, -0x190(s0)
80003480: a009         	j	0x80003482 <mf_conv2d_3x3_i8+0x3e4>
80003482: e7043503     	ld	a0, -0x190(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:227
;                 int k_valid = MF_MIN(tile_k, K - k_start);
80003486: eea42223     	sw	a0, -0x11c(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:230
;                 if (k_valid < tile_k || actual_n < tile_n)
8000348a: ee442503     	lw	a0, -0x11c(s0)
8000348e: 02000593     	li	a1, 0x20
80003492: 00b54a63     	blt	a0, a1, 0x800034a6 <mf_conv2d_3x3_i8+0x408>
80003496: a009         	j	0x80003498 <mf_conv2d_3x3_i8+0x3fa>
80003498: ef442583     	lw	a1, -0x10c(s0)
8000349c: 03f00513     	li	a0, 0x3f
800034a0: 00b54c63     	blt	a0, a1, 0x800034b8 <mf_conv2d_3x3_i8+0x41a>
800034a4: a009         	j	0x800034a6 <mf_conv2d_3x3_i8+0x408>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:231
;                     mf_memset(b_tmp, 0, tile_k * tile_n);
800034a6: f4043503     	ld	a0, -0xc0(s0)
800034aa: 4585         	li	a1, 0x1
800034ac: 00b59613     	slli	a2, a1, 0xb
800034b0: 4581         	li	a1, 0x0
800034b2: b80ff0ef     	jal	0x80002832 <mf_memset>
800034b6: a009         	j	0x800034b8 <mf_conv2d_3x3_i8+0x41a>
800034b8: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:233
;                 for (int ki = 0; ki < k_valid; ki++) {
800034ba: eea42023     	sw	a0, -0x120(s0)
800034be: a009         	j	0x800034c0 <mf_conv2d_3x3_i8+0x422>
800034c0: ee042503     	lw	a0, -0x120(s0)
800034c4: ee442583     	lw	a1, -0x11c(s0)
800034c8: 0cb55863     	bge	a0, a1, 0x80003598 <mf_conv2d_3x3_i8+0x4fa>
800034cc: a009         	j	0x800034ce <mf_conv2d_3x3_i8+0x430>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:234
;                     int k  = k_start + ki;
800034ce: ee842503     	lw	a0, -0x118(s0)
800034d2: ee042583     	lw	a1, -0x120(s0)
800034d6: 9d2d         	addw	a0, a0, a1
800034d8: eca42e23     	sw	a0, -0x124(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:235
;                     int ci = k / (kH * kW);
800034dc: edc42503     	lw	a0, -0x124(s0)
800034e0: 38e395b7     	lui	a1, 0x38e39
800034e4: e3958593     	addi	a1, a1, -0x1c7
800034e8: 02b50533     	mul	a0, a0, a1
800034ec: 03f55613     	srli	a2, a0, 0x3f
800034f0: 9505         	srai	a0, a0, 0x21
800034f2: 9d31         	addw	a0, a0, a2
800034f4: eca42c23     	sw	a0, -0x128(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:236
;                     int kh = (k % (kH * kW)) / kW;
800034f8: edc42503     	lw	a0, -0x124(s0)
800034fc: 02b505b3     	mul	a1, a0, a1
80003500: 03f5d613     	srli	a2, a1, 0x3f
80003504: 9581         	srai	a1, a1, 0x20
80003506: 8185         	srli	a1, a1, 0x1
80003508: 9e2d         	addw	a2, a2, a1
8000350a: 0036159b     	slliw	a1, a2, 0x3
8000350e: 9db1         	addw	a1, a1, a2
80003510: 9d0d         	subw	a0, a0, a1
80003512: 555555b7     	lui	a1, 0x55555
80003516: 55658593     	addi	a1, a1, 0x556
8000351a: 02b50533     	mul	a0, a0, a1
8000351e: 03f55613     	srli	a2, a0, 0x3f
80003522: 9101         	srli	a0, a0, 0x20
80003524: 9d31         	addw	a0, a0, a2
80003526: eca42a23     	sw	a0, -0x12c(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:237
;                     int kw = k % kW;
8000352a: edc42503     	lw	a0, -0x124(s0)
8000352e: 02b505b3     	mul	a1, a0, a1
80003532: 03f5d613     	srli	a2, a1, 0x3f
80003536: 9181         	srli	a1, a1, 0x20
80003538: 9e2d         	addw	a2, a2, a1
8000353a: 0016159b     	slliw	a1, a2, 0x1
8000353e: 9db1         	addw	a1, a1, a2
80003540: 9d0d         	subw	a0, a0, a1
80003542: eca42823     	sw	a0, -0x130(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:238
;                     mf_btile_fill_s1_i8((int8_t *)b_tmp + ki * tile_n,
80003546: f4043503     	ld	a0, -0xc0(s0)
8000354a: ee042583     	lw	a1, -0x120(s0)
8000354e: 0065959b     	slliw	a1, a1, 0x6
80003552: 952e         	add	a0, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:239
;                                         padded, ci, kh, kw,
80003554: f5043583     	ld	a1, -0xb0(s0)
80003558: ed842603     	lw	a2, -0x128(s0)
8000355c: ed442683     	lw	a3, -0x12c(s0)
80003560: ed042703     	lw	a4, -0x130(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:240
;                                         padH, padW,
80003564: f7c42783     	lw	a5, -0x84(s0)
80003568: f7842803     	lw	a6, -0x88(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:241
;                                         n_start, actual_n, outW);
8000356c: ef842883     	lw	a7, -0x108(s0)
80003570: ef442283     	lw	t0, -0x10c(s0)
80003574: f9c42383     	lw	t2, -0x64(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:238
;                     mf_btile_fill_s1_i8((int8_t *)b_tmp + ki * tile_n,
80003578: 1141         	addi	sp, sp, -0x10
8000357a: 830a         	mv	t1, sp
8000357c: 00733423     	sd	t2, 0x8(t1)
80003580: 00533023     	sd	t0, 0x0(t1)
80003584: 2a8000ef     	jal	0x8000382c <mf_btile_fill_s1_i8>
80003588: 0141         	addi	sp, sp, 0x10
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:242
;                 }
8000358a: a009         	j	0x8000358c <mf_conv2d_3x3_i8+0x4ee>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:233
;                 for (int ki = 0; ki < k_valid; ki++) {
8000358c: ee042503     	lw	a0, -0x120(s0)
80003590: 2505         	addiw	a0, a0, 0x1
80003592: eea42023     	sw	a0, -0x120(s0)
80003596: b72d         	j	0x800034c0 <mf_conv2d_3x3_i8+0x422>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:243
;                 __builtin_riscv_mf_dma_sync();
80003598: 8000000b     	dma.sync
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:246
;                 mf_dma_transpose_8(b_pad, b_tmp,
8000359c: f3843503     	ld	a0, -0xc8(s0)
800035a0: f4043583     	ld	a1, -0xc0(s0)
800035a4: 04000693     	li	a3, 0x40
800035a8: 02000713     	li	a4, 0x20
800035ac: 8636         	mv	a2, a3
800035ae: 98bff0ef     	jal	0x80002f38 <mf_dma_transpose_8>
800035b2: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:252
;                 for (int a = 0; a < m_batch; a++) {
800035b4: eca42623     	sw	a0, -0x134(s0)
800035b8: a009         	j	0x800035ba <mf_conv2d_3x3_i8+0x51c>
800035ba: ecc42503     	lw	a0, -0x134(s0)
800035be: f0042583     	lw	a1, -0x100(s0)
800035c2: 08b55a63     	bge	a0, a1, 0x80003656 <mf_conv2d_3x3_i8+0x5b8>
800035c6: a009         	j	0x800035c8 <mf_conv2d_3x3_i8+0x52a>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:253
;                     int ms = (mg + a) * tile_m;
800035c8: f0442503     	lw	a0, -0xfc(s0)
800035cc: ecc42583     	lw	a1, -0x134(s0)
800035d0: 9d2d         	addw	a0, a0, a1
800035d2: 0045151b     	slliw	a0, a0, 0x4
800035d6: eca42423     	sw	a0, -0x138(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:254
;                     int ml = MF_MIN(tile_m, C_out - ms);
800035da: fb842503     	lw	a0, -0x48(s0)
800035de: ec842583     	lw	a1, -0x138(s0)
800035e2: 9d0d         	subw	a0, a0, a1
800035e4: 45c5         	li	a1, 0x11
800035e6: 00b54763     	blt	a0, a1, 0x800035f4 <mf_conv2d_3x3_i8+0x556>
800035ea: a009         	j	0x800035ec <mf_conv2d_3x3_i8+0x54e>
800035ec: 4541         	li	a0, 0x10
800035ee: e6a43423     	sd	a0, -0x198(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:254
;                     int ml = MF_MIN(tile_m, C_out - ms);
800035f2: a809         	j	0x80003604 <mf_conv2d_3x3_i8+0x566>
800035f4: fb842503     	lw	a0, -0x48(s0)
800035f8: ec842583     	lw	a1, -0x138(s0)
800035fc: 9d0d         	subw	a0, a0, a1
800035fe: e6a43423     	sd	a0, -0x198(s0)
80003602: a009         	j	0x80003604 <mf_conv2d_3x3_i8+0x566>
80003604: e6843503     	ld	a0, -0x198(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:254
;                     int ml = MF_MIN(tile_m, C_out - ms);
80003608: eca42223     	sw	a0, -0x13c(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:255
;                     mf_atile_load(a_pad,
8000360c: f4843503     	ld	a0, -0xb8(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:256
;                                   w_base + ms * K_padded + k_start,
80003610: f1043583     	ld	a1, -0xf0(s0)
80003614: ec842603     	lw	a2, -0x138(s0)
80003618: f9442803     	lw	a6, -0x6c(s0)
8000361c: 0306063b     	mulw	a2, a2, a6
80003620: 95b2         	add	a1, a1, a2
80003622: ee842603     	lw	a2, -0x118(s0)
80003626: 95b2         	add	a1, a1, a2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:257
;                                   ml, tile_m, tile_k, tile_k, K_padded);
80003628: ec442603     	lw	a2, -0x13c(s0)
8000362c: 46c1         	li	a3, 0x10
8000362e: 02000793     	li	a5, 0x20
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:255
;                     mf_atile_load(a_pad,
80003632: 873e         	mv	a4, a5
80003634: 324000ef     	jal	0x80003958 <mf_atile_load>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:258
;                     mf_tmma_ii8(a, a_pad, b_pad);
80003638: ecc42503     	lw	a0, -0x134(s0)
8000363c: f4843583     	ld	a1, -0xb8(s0)
80003640: f3843603     	ld	a2, -0xc8(s0)
80003644: 93fff0ef     	jal	0x80002f82 <mf_tmma_ii8>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:259
;                 }
80003648: a009         	j	0x8000364a <mf_conv2d_3x3_i8+0x5ac>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:252
;                 for (int a = 0; a < m_batch; a++) {
8000364a: ecc42503     	lw	a0, -0x134(s0)
8000364e: 2505         	addiw	a0, a0, 0x1
80003650: eca42623     	sw	a0, -0x134(s0)
80003654: b79d         	j	0x800035ba <mf_conv2d_3x3_i8+0x51c>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:260
;             }
80003656: a009         	j	0x80003658 <mf_conv2d_3x3_i8+0x5ba>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:225
;             for (int kt = 0; kt < k_tiles; kt++) {
80003658: eec42503     	lw	a0, -0x114(s0)
8000365c: 2505         	addiw	a0, a0, 0x1
8000365e: eea42623     	sw	a0, -0x114(s0)
80003662: bbe1         	j	0x8000343a <mf_conv2d_3x3_i8+0x39c>
80003664: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:263
;             for (int a = 0; a < m_batch; a++) {
80003666: eca42023     	sw	a0, -0x140(s0)
8000366a: a009         	j	0x8000366c <mf_conv2d_3x3_i8+0x5ce>
8000366c: ec042503     	lw	a0, -0x140(s0)
80003670: f0042583     	lw	a1, -0x100(s0)
80003674: 0cb55d63     	bge	a0, a1, 0x8000374e <mf_conv2d_3x3_i8+0x6b0>
80003678: a009         	j	0x8000367a <mf_conv2d_3x3_i8+0x5dc>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:264
;                 int ms = (mg + a) * tile_m;
8000367a: f0442503     	lw	a0, -0xfc(s0)
8000367e: ec042583     	lw	a1, -0x140(s0)
80003682: 9d2d         	addw	a0, a0, a1
80003684: 0045151b     	slliw	a0, a0, 0x4
80003688: eaa42e23     	sw	a0, -0x144(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:265
;                 int ml = MF_MIN(tile_m, C_out - ms);
8000368c: fb842503     	lw	a0, -0x48(s0)
80003690: ebc42583     	lw	a1, -0x144(s0)
80003694: 9d0d         	subw	a0, a0, a1
80003696: 45c5         	li	a1, 0x11
80003698: 00b54763     	blt	a0, a1, 0x800036a6 <mf_conv2d_3x3_i8+0x608>
8000369c: a009         	j	0x8000369e <mf_conv2d_3x3_i8+0x600>
8000369e: 4541         	li	a0, 0x10
800036a0: e6a43023     	sd	a0, -0x1a0(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:265
;                 int ml = MF_MIN(tile_m, C_out - ms);
800036a4: a809         	j	0x800036b6 <mf_conv2d_3x3_i8+0x618>
800036a6: fb842503     	lw	a0, -0x48(s0)
800036aa: ebc42583     	lw	a1, -0x144(s0)
800036ae: 9d0d         	subw	a0, a0, a1
800036b0: e6a43023     	sd	a0, -0x1a0(s0)
800036b4: a009         	j	0x800036b6 <mf_conv2d_3x3_i8+0x618>
800036b6: e6043503     	ld	a0, -0x1a0(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:265
;                 int ml = MF_MIN(tile_m, C_out - ms);
800036ba: eaa42c23     	sw	a0, -0x148(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:266
;                 uint8_t *c_dst = c_base + (ms * N + n_start) * c_elem;
800036be: f0843503     	ld	a0, -0xf8(s0)
800036c2: ebc42583     	lw	a1, -0x144(s0)
800036c6: f9042603     	lw	a2, -0x70(s0)
800036ca: 02c585bb     	mulw	a1, a1, a2
800036ce: ef842603     	lw	a2, -0x108(s0)
800036d2: 9db1         	addw	a1, a1, a2
800036d4: 0025959b     	slliw	a1, a1, 0x2
800036d8: 952e         	add	a0, a0, a1
800036da: eaa43823     	sd	a0, -0x150(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:268
;                 mf_tstore32(a, c_tmp,
800036de: ec042503     	lw	a0, -0x140(s0)
800036e2: f3043583     	ld	a1, -0xd0(s0)
800036e6: 6605         	lui	a2, 0x1
800036e8: 929ff0ef     	jal	0x80003010 <mf_tstore32>
800036ec: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:270
;                 for (int r = 0; r < ml; r++)
800036ee: eaa42623     	sw	a0, -0x154(s0)
800036f2: a009         	j	0x800036f4 <mf_conv2d_3x3_i8+0x656>
800036f4: eac42503     	lw	a0, -0x154(s0)
800036f8: eb842583     	lw	a1, -0x148(s0)
800036fc: 04b55063     	bge	a0, a1, 0x8000373c <mf_conv2d_3x3_i8+0x69e>
80003700: a009         	j	0x80003702 <mf_conv2d_3x3_i8+0x664>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:272
;                         c_dst + r * N * c_elem,
80003702: eb043503     	ld	a0, -0x150(s0)
80003706: eac42603     	lw	a2, -0x154(s0)
8000370a: f9042583     	lw	a1, -0x70(s0)
8000370e: 02b605bb     	mulw	a1, a2, a1
80003712: 0025959b     	slliw	a1, a1, 0x2
80003716: 952e         	add	a0, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:273
;                         (void *)(c_tmp + r * tile_n * c_elem),
80003718: f3043583     	ld	a1, -0xd0(s0)
8000371c: 0086161b     	slliw	a2, a2, 0x8
80003720: 95b2         	add	a1, a1, a2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:274
;                         (unsigned long)(actual_n * c_elem));
80003722: ef442603     	lw	a2, -0x10c(s0)
80003726: 0026161b     	slliw	a2, a2, 0x2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:271
;                     __builtin_riscv_mf_dma_x(
8000372a: 00c5d50b     	dma.x	a0, a1, a2
8000372e: a009         	j	0x80003730 <mf_conv2d_3x3_i8+0x692>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:270
;                 for (int r = 0; r < ml; r++)
80003730: eac42503     	lw	a0, -0x154(s0)
80003734: 2505         	addiw	a0, a0, 0x1
80003736: eaa42623     	sw	a0, -0x154(s0)
8000373a: bf6d         	j	0x800036f4 <mf_conv2d_3x3_i8+0x656>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:275
;                 __builtin_riscv_mf_dma_sync();
8000373c: 8000000b     	dma.sync
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:276
;             }
80003740: a009         	j	0x80003742 <mf_conv2d_3x3_i8+0x6a4>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:263
;             for (int a = 0; a < m_batch; a++) {
80003742: ec042503     	lw	a0, -0x140(s0)
80003746: 2505         	addiw	a0, a0, 0x1
80003748: eca42023     	sw	a0, -0x140(s0)
8000374c: b705         	j	0x8000366c <mf_conv2d_3x3_i8+0x5ce>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:277
;         }
8000374e: a009         	j	0x80003750 <mf_conv2d_3x3_i8+0x6b2>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:218
;         for (int nt = 0; nt < n_tiles; nt++) {
80003750: efc42503     	lw	a0, -0x104(s0)
80003754: 2505         	addiw	a0, a0, 0x1
80003756: eea42e23     	sw	a0, -0x104(s0)
8000375a: b9b1         	j	0x800033b6 <mf_conv2d_3x3_i8+0x318>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:278
;     }
8000375c: a009         	j	0x8000375e <mf_conv2d_3x3_i8+0x6c0>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:215
;     for (int mg = 0; mg < m_tiles; mg += MF_NUM_ACC) {
8000375e: f0442503     	lw	a0, -0xfc(s0)
80003762: 2511         	addiw	a0, a0, 0x4
80003764: f0a42223     	sw	a0, -0xfc(s0)
80003768: b119         	j	0x8000336e <mf_conv2d_3x3_i8+0x2d0>
8000376a: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:279
;     return MF_OK;
8000376c: fea42623     	sw	a0, -0x14(s0)
80003770: a009         	j	0x80003772 <mf_conv2d_3x3_i8+0x6d4>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:280
; }
80003772: fec42503     	lw	a0, -0x14(s0)
80003776: e6040113     	addi	sp, s0, -0x1a0
8000377a: 60fa         	ld	ra, 0x198(sp)
8000377c: 645a         	ld	s0, 0x190(sp)
8000377e: 611d         	addi	sp, sp, 0x1a0
80003780: 8082         	ret

0000000080003782 <mf_l1_alloc_init>:
; mf_l1_alloc_init():
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:158
; static inline void mf_l1_alloc_init(mf_l1_alloc_t *a, mf_l1_buf_t ws) {
80003782: 7179         	addi	sp, sp, -0x30
80003784: f406         	sd	ra, 0x28(sp)
80003786: f022         	sd	s0, 0x20(sp)
80003788: 1800         	addi	s0, sp, 0x30
8000378a: fec43423     	sd	a2, -0x18(s0)
8000378e: feb43023     	sd	a1, -0x20(s0)
80003792: fca43c23     	sd	a0, -0x28(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:159
;     a->base     = (uint8_t *)ws.ptr;
80003796: fe043503     	ld	a0, -0x20(s0)
8000379a: fd843583     	ld	a1, -0x28(s0)
8000379e: e188         	sd	a0, 0x0(a1)
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:160
;     a->offset   = 0;
800037a0: fd843583     	ld	a1, -0x28(s0)
800037a4: 4501         	li	a0, 0x0
800037a6: e588         	sd	a0, 0x8(a1)
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:161
;     a->capacity = ws.size;
800037a8: fe843503     	ld	a0, -0x18(s0)
800037ac: fd843583     	ld	a1, -0x28(s0)
800037b0: e988         	sd	a0, 0x10(a1)
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:162
; }
800037b2: fd040113     	addi	sp, s0, -0x30
800037b6: 70a2         	ld	ra, 0x28(sp)
800037b8: 7402         	ld	s0, 0x20(sp)
800037ba: 6145         	addi	sp, sp, 0x30
800037bc: 8082         	ret

00000000800037be <mf_l1_alloc>:
; mf_l1_alloc():
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:164
; static inline void *mf_l1_alloc(mf_l1_alloc_t *a, size_t bytes) {
800037be: 7179         	addi	sp, sp, -0x30
800037c0: f406         	sd	ra, 0x28(sp)
800037c2: f022         	sd	s0, 0x20(sp)
800037c4: 1800         	addi	s0, sp, 0x30
800037c6: fea43023     	sd	a0, -0x20(s0)
800037ca: fcb43c23     	sd	a1, -0x28(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:165
;     bytes = MF_ALIGN_UP(bytes, 8);
800037ce: fd843503     	ld	a0, -0x28(s0)
800037d2: 051d         	addi	a0, a0, 0x7
800037d4: 9961         	andi	a0, a0, -0x8
800037d6: fca43c23     	sd	a0, -0x28(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:166
;     if (a->offset + bytes > a->capacity)
800037da: fe043503     	ld	a0, -0x20(s0)
800037de: 650c         	ld	a1, 0x8(a0)
800037e0: fd843603     	ld	a2, -0x28(s0)
800037e4: 95b2         	add	a1, a1, a2
800037e6: 6908         	ld	a0, 0x10(a0)
800037e8: 00b57763     	bgeu	a0, a1, 0x800037f6 <mf_l1_alloc+0x38>
800037ec: a009         	j	0x800037ee <mf_l1_alloc+0x30>
800037ee: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:167
;         return NULL;
800037f0: fea43423     	sd	a0, -0x18(s0)
800037f4: a025         	j	0x8000381c <mf_l1_alloc+0x5e>
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:168
;     void *p = a->base + a->offset;
800037f6: fe043583     	ld	a1, -0x20(s0)
800037fa: 6188         	ld	a0, 0x0(a1)
800037fc: 658c         	ld	a1, 0x8(a1)
800037fe: 952e         	add	a0, a0, a1
80003800: fca43823     	sd	a0, -0x30(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:169
;     a->offset += bytes;
80003804: fd843603     	ld	a2, -0x28(s0)
80003808: fe043583     	ld	a1, -0x20(s0)
8000380c: 6588         	ld	a0, 0x8(a1)
8000380e: 9532         	add	a0, a0, a2
80003810: e588         	sd	a0, 0x8(a1)
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:170
;     return p;
80003812: fd043503     	ld	a0, -0x30(s0)
80003816: fea43423     	sd	a0, -0x18(s0)
8000381a: a009         	j	0x8000381c <mf_l1_alloc+0x5e>
; /home/heyi/heyi/matrixflow/kernel/./mf_kernel.h:171
; }
8000381c: fe843503     	ld	a0, -0x18(s0)
80003820: fd040113     	addi	sp, s0, -0x30
80003824: 70a2         	ld	ra, 0x28(sp)
80003826: 7402         	ld	s0, 0x20(sp)
80003828: 6145         	addi	sp, sp, 0x30
8000382a: 8082         	ret

000000008000382c <mf_btile_fill_s1_i8>:
; mf_btile_fill_s1_i8():
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:55
;                                          int outW) {
8000382c: 711d         	addi	sp, sp, -0x60
8000382e: ec86         	sd	ra, 0x58(sp)
80003830: e8a2         	sd	s0, 0x50(sp)
80003832: 1080         	addi	s0, sp, 0x60
80003834: 82ae         	mv	t0, a1
80003836: 832a         	mv	t1, a0
80003838: 6408         	ld	a0, 0x8(s0)
8000383a: 600c         	ld	a1, 0x0(s0)
8000383c: fe643423     	sd	t1, -0x18(s0)
80003840: fe543023     	sd	t0, -0x20(s0)
80003844: fcc42e23     	sw	a2, -0x24(s0)
80003848: fcd42c23     	sw	a3, -0x28(s0)
8000384c: fce42a23     	sw	a4, -0x2c(s0)
80003850: fcf42823     	sw	a5, -0x30(s0)
80003854: fd042623     	sw	a6, -0x34(s0)
80003858: fd142423     	sw	a7, -0x38(s0)
8000385c: fcb42223     	sw	a1, -0x3c(s0)
80003860: fca42023     	sw	a0, -0x40(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:56
;     int remaining = actual_n;
80003864: fc442503     	lw	a0, -0x3c(s0)
80003868: faa42e23     	sw	a0, -0x44(s0)
8000386c: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:57
;     int b_col = 0;
8000386e: faa42c23     	sw	a0, -0x48(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:58
;     int cur_oh = n_start / outW;
80003872: fc842503     	lw	a0, -0x38(s0)
80003876: fc042583     	lw	a1, -0x40(s0)
8000387a: 02b5453b     	divw	a0, a0, a1
8000387e: faa42a23     	sw	a0, -0x4c(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:59
;     int cur_ow = n_start % outW;
80003882: fc842503     	lw	a0, -0x38(s0)
80003886: fc042583     	lw	a1, -0x40(s0)
8000388a: 02b5653b     	remw	a0, a0, a1
8000388e: faa42823     	sw	a0, -0x50(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:61
;     while (remaining > 0) {
80003892: a009         	j	0x80003894 <mf_btile_fill_s1_i8+0x68>
80003894: fbc42583     	lw	a1, -0x44(s0)
80003898: 4501         	li	a0, 0x0
8000389a: 0ab55963     	bge	a0, a1, 0x8000394c <mf_btile_fill_s1_i8+0x120>
8000389e: a009         	j	0x800038a0 <mf_btile_fill_s1_i8+0x74>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:62
;         int run = MF_MIN(remaining, outW - cur_ow);
800038a0: fbc42503     	lw	a0, -0x44(s0)
800038a4: fc042583     	lw	a1, -0x40(s0)
800038a8: fb042603     	lw	a2, -0x50(s0)
800038ac: 9d91         	subw	a1, a1, a2
800038ae: 00b55863     	bge	a0, a1, 0x800038be <mf_btile_fill_s1_i8+0x92>
800038b2: a009         	j	0x800038b4 <mf_btile_fill_s1_i8+0x88>
800038b4: fbc42503     	lw	a0, -0x44(s0)
800038b8: faa43023     	sd	a0, -0x60(s0)
800038bc: a809         	j	0x800038ce <mf_btile_fill_s1_i8+0xa2>
800038be: fc042503     	lw	a0, -0x40(s0)
800038c2: fb042583     	lw	a1, -0x50(s0)
800038c6: 9d0d         	subw	a0, a0, a1
800038c8: faa43023     	sd	a0, -0x60(s0)
800038cc: a009         	j	0x800038ce <mf_btile_fill_s1_i8+0xa2>
800038ce: fa043503     	ld	a0, -0x60(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:62
;         int run = MF_MIN(remaining, outW - cur_ow);
800038d2: faa42623     	sw	a0, -0x54(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:64
;             b_row + b_col,
800038d6: fe843503     	ld	a0, -0x18(s0)
800038da: fb842583     	lw	a1, -0x48(s0)
800038de: 952e         	add	a0, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:65
;             (void *)(padded + ci * padH * padW
800038e0: fe043583     	ld	a1, -0x20(s0)
800038e4: fdc42603     	lw	a2, -0x24(s0)
800038e8: fd042683     	lw	a3, -0x30(s0)
800038ec: 02d6063b     	mulw	a2, a2, a3
800038f0: fcc42683     	lw	a3, -0x34(s0)
800038f4: 02d6063b     	mulw	a2, a2, a3
800038f8: 95b2         	add	a1, a1, a2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:66
;                      + (cur_oh + kh) * padW
800038fa: fb442603     	lw	a2, -0x4c(s0)
800038fe: fd842703     	lw	a4, -0x28(s0)
80003902: 9e39         	addw	a2, a2, a4
80003904: 02d6063b     	mulw	a2, a2, a3
80003908: 95b2         	add	a1, a1, a2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:67
;                      + (cur_ow + kw)),
8000390a: fb042603     	lw	a2, -0x50(s0)
8000390e: fd442683     	lw	a3, -0x2c(s0)
80003912: 9e35         	addw	a2, a2, a3
80003914: 95b2         	add	a1, a1, a2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:68
;             (unsigned long)run);
80003916: fac42603     	lw	a2, -0x54(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:63
;         __builtin_riscv_mf_dma_x(
8000391a: 00c5d50b     	dma.x	a0, a1, a2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:69
;         b_col += run;
8000391e: fac42583     	lw	a1, -0x54(s0)
80003922: fb842503     	lw	a0, -0x48(s0)
80003926: 9d2d         	addw	a0, a0, a1
80003928: faa42c23     	sw	a0, -0x48(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:70
;         remaining -= run;
8000392c: fac42583     	lw	a1, -0x54(s0)
80003930: fbc42503     	lw	a0, -0x44(s0)
80003934: 9d0d         	subw	a0, a0, a1
80003936: faa42e23     	sw	a0, -0x44(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:71
;         cur_oh++;
8000393a: fb442503     	lw	a0, -0x4c(s0)
8000393e: 2505         	addiw	a0, a0, 0x1
80003940: faa42a23     	sw	a0, -0x4c(s0)
80003944: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:72
;         cur_ow = 0;
80003946: faa42823     	sw	a0, -0x50(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:61
;     while (remaining > 0) {
8000394a: b7a9         	j	0x80003894 <mf_btile_fill_s1_i8+0x68>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:74
; }
8000394c: fa040113     	addi	sp, s0, -0x60
80003950: 60e6         	ld	ra, 0x58(sp)
80003952: 6446         	ld	s0, 0x50(sp)
80003954: 6125         	addi	sp, sp, 0x60
80003956: 8082         	ret

0000000080003958 <mf_atile_load>:
; mf_atile_load():
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:117
;                                    int w_stride_bytes) {
80003958: 7139         	addi	sp, sp, -0x40
8000395a: fc06         	sd	ra, 0x38(sp)
8000395c: f822         	sd	s0, 0x30(sp)
8000395e: 0080         	addi	s0, sp, 0x40
80003960: fea43423     	sd	a0, -0x18(s0)
80003964: feb43023     	sd	a1, -0x20(s0)
80003968: fcc42e23     	sw	a2, -0x24(s0)
8000396c: fcd42c23     	sw	a3, -0x28(s0)
80003970: fce42a23     	sw	a4, -0x2c(s0)
80003974: fcf42823     	sw	a5, -0x30(s0)
80003978: fd042623     	sw	a6, -0x34(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:118
;     if (ml < tile_m || load_k_bytes < tile_k_bytes)
8000397c: fdc42503     	lw	a0, -0x24(s0)
80003980: fd842583     	lw	a1, -0x28(s0)
80003984: 00b54a63     	blt	a0, a1, 0x80003998 <mf_atile_load+0x40>
80003988: a009         	j	0x8000398a <mf_atile_load+0x32>
8000398a: fd442503     	lw	a0, -0x2c(s0)
8000398e: fd042583     	lw	a1, -0x30(s0)
80003992: 00b55f63     	bge	a0, a1, 0x800039b0 <mf_atile_load+0x58>
80003996: a009         	j	0x80003998 <mf_atile_load+0x40>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:119
;         mf_memset(a_pad, 0, tile_m * tile_k_bytes);
80003998: fe843503     	ld	a0, -0x18(s0)
8000399c: fd842583     	lw	a1, -0x28(s0)
800039a0: fd042603     	lw	a2, -0x30(s0)
800039a4: 02c5863b     	mulw	a2, a1, a2
800039a8: 4581         	li	a1, 0x0
800039aa: e89fe0ef     	jal	0x80002832 <mf_memset>
800039ae: a009         	j	0x800039b0 <mf_atile_load+0x58>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:121
;     if (load_k_bytes == tile_k_bytes) {
800039b0: fd442503     	lw	a0, -0x2c(s0)
800039b4: fd042583     	lw	a1, -0x30(s0)
800039b8: 02b51263     	bne	a0, a1, 0x800039dc <mf_atile_load+0x84>
800039bc: a009         	j	0x800039be <mf_atile_load+0x66>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:123
;         mf_dma_load_tile_2d(a_pad, w_src,
800039be: fe843503     	ld	a0, -0x18(s0)
800039c2: fe043583     	ld	a1, -0x20(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:124
;                             (size_t)tile_k_bytes,
800039c6: fd042603     	lw	a2, -0x30(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:125
;                             (size_t)ml,
800039ca: fdc42683     	lw	a3, -0x24(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:126
;                             (size_t)w_stride_bytes);
800039ce: fcc42703     	lw	a4, -0x34(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:123
;         mf_dma_load_tile_2d(a_pad, w_src,
800039d2: cdeff0ef     	jal	0x80002eb0 <mf_dma_load_tile_2d>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:127
;         mf_dma_sync();
800039d6: d4aff0ef     	jal	0x80002f20 <mf_dma_sync>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:128
;     } else {
800039da: a891         	j	0x80003a2e <mf_atile_load+0xd6>
800039dc: 4501         	li	a0, 0x0
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:130
;         for (int r = 0; r < ml; r++)
800039de: fca42423     	sw	a0, -0x38(s0)
800039e2: a009         	j	0x800039e4 <mf_atile_load+0x8c>
800039e4: fc842503     	lw	a0, -0x38(s0)
800039e8: fdc42583     	lw	a1, -0x24(s0)
800039ec: 02b55e63     	bge	a0, a1, 0x80003a28 <mf_atile_load+0xd0>
800039f0: a009         	j	0x800039f2 <mf_atile_load+0x9a>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:132
;                 a_pad + r * tile_k_bytes,
800039f2: fe843503     	ld	a0, -0x18(s0)
800039f6: fc842603     	lw	a2, -0x38(s0)
800039fa: fd042583     	lw	a1, -0x30(s0)
800039fe: 02b605bb     	mulw	a1, a2, a1
80003a02: 952e         	add	a0, a0, a1
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:133
;                 (void *)(w_src + r * w_stride_bytes),
80003a04: fe043583     	ld	a1, -0x20(s0)
80003a08: fcc42683     	lw	a3, -0x34(s0)
80003a0c: 02d6063b     	mulw	a2, a2, a3
80003a10: 95b2         	add	a1, a1, a2
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:134
;                 (unsigned long)load_k_bytes);
80003a12: fd442603     	lw	a2, -0x2c(s0)
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:131
;             __builtin_riscv_mf_dma_x(
80003a16: 00c5d50b     	dma.x	a0, a1, a2
80003a1a: a009         	j	0x80003a1c <mf_atile_load+0xc4>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:130
;         for (int r = 0; r < ml; r++)
80003a1c: fc842503     	lw	a0, -0x38(s0)
80003a20: 2505         	addiw	a0, a0, 0x1
80003a22: fca42423     	sw	a0, -0x38(s0)
80003a26: bf7d         	j	0x800039e4 <mf_atile_load+0x8c>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:135
;         __builtin_riscv_mf_dma_sync();
80003a28: 8000000b     	dma.sync
80003a2c: a009         	j	0x80003a2e <mf_atile_load+0xd6>
; /home/heyi/heyi/matrixflow/kernel/./mf_conv2d.h:137
; }
80003a2e: fc040113     	addi	sp, s0, -0x40
80003a32: 70e2         	ld	ra, 0x38(sp)
80003a34: 7442         	ld	s0, 0x30(sp)
80003a36: 6121         	addi	sp, sp, 0x40
80003a38: 8082         	ret
