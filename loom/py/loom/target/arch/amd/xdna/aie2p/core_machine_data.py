# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

"""Owned AIE2P physical-register and machine-form facts.

The compact records normalize the physical llvm-aie model at the revision
pinned by :mod:`core_encoding_data`. LLVM-AIE remains an extraction and
differential-testing oracle only; normal Loom builds have no dependency on it.
The three q-register adapters retain their source-compiler values and carry a
decoder-derived architectural override because llvm-aie's emitter aliases q0/q2
and q1/q3 at every two-bit q-register field.
"""

from __future__ import annotations

import json

from loom.target.arch.amd.xdna.aie.machine import (
    ImmediateEncoding,
    MachineForm,
    MachineOperand,
    MachineOperandKind,
    MachineTable,
    MachineTie,
    PhysicalRegister,
    RegisterAdapter,
    RegisterClass,
    RegisterLayout,
    validate_machine_table,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import (
    CORE_ENCODING_TABLE,
)

_ATOMIC_UNIT_NAME_RECORDS = """\
CORE_ID
bmhh0
bmhh1
bmhh2
bmhh3
bmhh4
bmhl0
bmhl1
bmhl2
bmhl3
bmhl4
bmlh0
bmlh1
bmlh2
bmlh3
bmlh4
bmll0
bmll1
bmll2
bmll3
bmll4
crF2BMask
crF2FMask
crF2IMask
crFPCnvFl2FxMask
crFPCnvFx2FlMask
crFPMask
crFPNlfMask
crMCDEn
crPackSize
crRnd
crSCDEn
crSRSMode
crSat
crUPSMode
crUnpackSize
dc0
dc1
dc2
dc3
dc4
dc5
dc6
dc7
dj0
dj1
dj2
dj3
dj4
dj5
dj6
dj7
dn0
dn1
dn2
dn3
dn4
dn5
dn6
dn7
eh0
eh1
eh10
eh11
eh2
eh3
eh4
eh5
eh6
eh7
eh8
eh9
el0
el1
el10
el11
el2
el3
el4
el5
el6
el7
el8
el9
lc
le
lfe
lfh0
lfh1
lfl0
lfl1
lr
ls
m0
m1
m2
m3
m4
m5
m6
m7
p0
p1
p2
p3
p4
p5
p6
p7
packSign0
packSign1
pe2_ads
qh0
qh1
qh2
qh3
ql0
ql1
ql2
ql3
r0
r1
r10
r11
r12
r13
r14
r15
r16
r17
r18
r19
r2
r20
r21
r22
r23
r24
r25
r26
r27
r28
r29
r3
r30
r31
r4
r5
r6
r7
r8
r9
s0
s1
s2
s3
sfh
sfl
sp
srCarry
srF2BFlags
srF2FFlags
srF2IFlags
srFPCnvFl2Fx
srFPCnvFx2Fl
srFPFlags
srFPNlf
srFifo_of
srFifo_uf
srMS0
srSRS_of
srSS0
srSparse_of
srUPS_of
srsSign0
srsSign1
tile_cntr
unpackSign0
unpackSign1
upsSign0
upsSign1
vaddSign0
vaddSign1
wh0
wh1
wh10
wh11
wh2
wh3
wh4
wh5
wh6
wh7
wh8
wh9
wl0
wl1
wl10
wl11
wl2
wl3
wl4
wl5
wl6
wl7
wl8
wl9
"""

_PHYSICAL_REGISTER_RECORDS = """\
CORE_ID|core_id|87|0|
bmhh0|bmhh0|0|1|
bmhh1|bmhh1|1|2|
bmhh2|bmhh2|2|3|
bmhh3|bmhh3|3|4|
bmhh4|bmhh4|4|5|
bmhl0|bmhl0|0|6|
bmhl1|bmhl1|1|7|
bmhl2|bmhl2|2|8|
bmhl3|bmhl3|3|9|
bmhl4|bmhl4|4|10|
bmlh0|bmlh0|0|11|
bmlh1|bmlh1|1|12|
bmlh2|bmlh2|2|13|
bmlh3|bmlh3|3|14|
bmlh4|bmlh4|4|15|
bmll0|bmll0|0|16|
bmll1|bmll1|1|17|
bmll2|bmll2|2|18|
bmll3|bmll3|3|19|
bmll4|bmll4|4|20|
cmh0|cmh0|0|1,6|bmhl0:sub_512_acc_lo,bmhh0:sub_512_acc_hi
cmh1|cmh1|1|2,7|bmhl1:sub_512_acc_lo,bmhh1:sub_512_acc_hi
cmh2|cmh2|2|3,8|bmhl2:sub_512_acc_lo,bmhh2:sub_512_acc_hi
cmh3|cmh3|3|4,9|bmhl3:sub_512_acc_lo,bmhh3:sub_512_acc_hi
cmh4|cmh4|4|5,10|bmhl4:sub_512_acc_lo,bmhh4:sub_512_acc_hi
cml0|cml0|0|11,16|bmll0:sub_512_acc_lo,bmlh0:sub_512_acc_hi
cml1|cml1|1|12,17|bmll1:sub_512_acc_lo,bmlh1:sub_512_acc_hi
cml2|cml2|2|13,18|bmll2:sub_512_acc_lo,bmlh2:sub_512_acc_hi
cml3|cml3|3|14,19|bmll3:sub_512_acc_lo,bmlh3:sub_512_acc_hi
cml4|cml4|4|15,20|bmll4:sub_512_acc_lo,bmlh4:sub_512_acc_hi
crF2BMask|crf2bmask|51|21|
crF2FMask|crf2fmask|115|22|
crF2IMask|crf2imask|11|23|
crFPCnvFl2FxMask|crfpcnvfl2fxmask|75|24|
crFPCnvFx2FlMask|crfpcnvfx2flmask|43|25|
crFPMask|crfpmask|107|26|
crFPNlfMask|crfpnlfmask|27|27|
crMCDEn|crmcden|91|28|
crPackSize|crpacksize|59|29|
crRnd|crrnd|123|30|
crSCDEn|crscden|7|31|
crSRSMode|crsrsmode|71|32|
crSat|crsat|39|33|
crUPSMode|crupsmode|103|34|
crUnpackSize|crunpacksize|23|35|
d0|d0|0|36,44,52,93|m0:sub_mod,dn0:sub_dim_size,dj0:sub_dim_stride,dc0:sub_dim_count
d0_3d|d0|0|36,40,44,48,52,56,93,97|d0:sub_lo_dim,d4:sub_hi_dim
d1|d1|1|37,45,53,94|m1:sub_mod,dn1:sub_dim_size,dj1:sub_dim_stride,dc1:sub_dim_count
d1_3d|d1|1|37,41,45,49,53,57,94,98|d1:sub_lo_dim,d5:sub_hi_dim
d2|d2|2|38,46,54,95|m2:sub_mod,dn2:sub_dim_size,dj2:sub_dim_stride,dc2:sub_dim_count
d2_3d|d2|2|38,42,46,50,54,58,95,99|d2:sub_lo_dim,d6:sub_hi_dim
d3|d3|3|39,47,55,96|m3:sub_mod,dn3:sub_dim_size,dj3:sub_dim_stride,dc3:sub_dim_count
d3_3d|d3|3|39,43,47,51,55,59,96,100|d3:sub_lo_dim,d7:sub_hi_dim
d4|d4|4|40,48,56,97|m4:sub_mod,dn4:sub_dim_size,dj4:sub_dim_stride,dc4:sub_dim_count
d5|d5|5|41,49,57,98|m5:sub_mod,dn5:sub_dim_size,dj5:sub_dim_stride,dc5:sub_dim_count
d6|d6|6|42,50,58,99|m6:sub_mod,dn6:sub_dim_size,dj6:sub_dim_stride,dc6:sub_dim_count
d7|d7|7|43,51,59,100|m7:sub_mod,dn7:sub_dim_size,dj7:sub_dim_stride,dc7:sub_dim_count
dc0|dc0|0|36|
dc1|dc1|1|37|
dc2|dc2|2|38|
dc3|dc3|3|39|
dc4|dc4|4|40|
dc5|dc5|5|41|
dc6|dc6|6|42|
dc7|dc7|7|43|
dj0|dj0|0|44|
dj1|dj1|1|45|
dj2|dj2|2|46|
dj3|dj3|3|47|
dj4|dj4|4|48|
dj5|dj5|5|49|
dj6|dj6|6|50|
dj7|dj7|7|51|
dm0|dm0|0|1,6,11,16|cml0:sub_1024_acc_lo,cmh0:sub_1024_acc_hi
dm1|dm1|1|2,7,12,17|cml1:sub_1024_acc_lo,cmh1:sub_1024_acc_hi
dm2|dm2|2|3,8,13,18|cml2:sub_1024_acc_lo,cmh2:sub_1024_acc_hi
dm3|dm3|3|4,9,14,19|cml3:sub_1024_acc_lo,cmh3:sub_1024_acc_hi
dm4|dm4|4|5,10,15,20|cml4:sub_1024_acc_lo,cmh4:sub_1024_acc_hi
dn0|dn0|0|52|
dn1|dn1|1|53|
dn2|dn2|2|54|
dn3|dn3|3|55|
dn4|dn4|4|56|
dn5|dn5|5|57|
dn6|dn6|6|58|
dn7|dn7|7|59|
e0|e0|0|60,72|el0:sub_lo_exp,eh0:sub_hi_exp
e1|e1|1|61,73|el1:sub_lo_exp,eh1:sub_hi_exp
e10|e10|10|62,74|el10:sub_lo_exp,eh10:sub_hi_exp
e11|e11|11|63,75|el11:sub_lo_exp,eh11:sub_hi_exp
e2|e2|2|64,76|el2:sub_lo_exp,eh2:sub_hi_exp
e3|e3|3|65,77|el3:sub_lo_exp,eh3:sub_hi_exp
e4|e4|4|66,78|el4:sub_lo_exp,eh4:sub_hi_exp
e5|e5|5|67,79|el5:sub_lo_exp,eh5:sub_hi_exp
e6|e6|6|68,80|el6:sub_lo_exp,eh6:sub_hi_exp
e7|e7|7|69,81|el7:sub_lo_exp,eh7:sub_hi_exp
e8|e8|8|70,82|el8:sub_lo_exp,eh8:sub_hi_exp
e9|e9|9|71,83|el9:sub_lo_exp,eh9:sub_hi_exp
eh0|eh0|0|60|
eh1|eh1|0|61|
eh10|eh10|5|62|
eh11|eh11|5|63|
eh2|eh2|1|64|
eh3|eh3|1|65|
eh4|eh4|2|66|
eh5|eh5|2|67|
eh6|eh6|3|68|
eh7|eh7|3|69|
eh8|eh8|4|70|
eh9|eh9|4|71|
el0|el0|0|72|
el1|el1|0|73|
el10|el10|5|74|
el11|el11|5|75|
el2|el2|1|76|
el3|el3|1|77|
el4|el4|2|78|
el5|el5|2|79|
el6|el6|3|80|
el7|el7|3|81|
el8|el8|4|82|
el9|el9|4|83|
ewh0|ewh0|0|60,183|eh0:sub_bfp_exp,wh0:sub_bfp_vec
ewh1|ewh1|0|61,184|eh1:sub_bfp_exp,wh1:sub_bfp_vec
ewh10|ewh10|5|62,185|eh10:sub_bfp_exp,wh10:sub_bfp_vec
ewh11|ewh11|5|63,186|eh11:sub_bfp_exp,wh11:sub_bfp_vec
ewh2|ewh2|1|64,187|eh2:sub_bfp_exp,wh2:sub_bfp_vec
ewh3|ewh3|1|65,188|eh3:sub_bfp_exp,wh3:sub_bfp_vec
ewh4|ewh4|2|66,189|eh4:sub_bfp_exp,wh4:sub_bfp_vec
ewh5|ewh5|2|67,190|eh5:sub_bfp_exp,wh5:sub_bfp_vec
ewh6|ewh6|3|68,191|eh6:sub_bfp_exp,wh6:sub_bfp_vec
ewh7|ewh7|3|69,192|eh7:sub_bfp_exp,wh7:sub_bfp_vec
ewh8|ewh8|4|70,193|eh8:sub_bfp_exp,wh8:sub_bfp_vec
ewh9|ewh9|4|71,194|eh9:sub_bfp_exp,wh9:sub_bfp_vec
ewl0|ewl0|0|72,195|el0:sub_bfp_exp,wl0:sub_bfp_vec
ewl1|ewl1|0|73,196|el1:sub_bfp_exp,wl1:sub_bfp_vec
ewl10|ewl10|5|74,197|el10:sub_bfp_exp,wl10:sub_bfp_vec
ewl11|ewl11|5|75,198|el11:sub_bfp_exp,wl11:sub_bfp_vec
ewl2|ewl2|1|76,199|el2:sub_bfp_exp,wl2:sub_bfp_vec
ewl3|ewl3|1|77,200|el3:sub_bfp_exp,wl3:sub_bfp_vec
ewl4|ewl4|2|78,201|el4:sub_bfp_exp,wl4:sub_bfp_vec
ewl5|ewl5|2|79,202|el5:sub_bfp_exp,wl5:sub_bfp_vec
ewl6|ewl6|3|80,203|el6:sub_bfp_exp,wl6:sub_bfp_vec
ewl7|ewl7|3|81,204|el7:sub_bfp_exp,wl7:sub_bfp_vec
ewl8|ewl8|4|82,205|el8:sub_bfp_exp,wl8:sub_bfp_vec
ewl9|ewl9|4|83,206|el9:sub_bfp_exp,wl9:sub_bfp_vec
ex0|ex0|0|60,72,183,195|x0:sub_bfp16_x,e0:sub_bfp16_e
ex1|ex1|0|61,73,184,196|x1:sub_bfp16_x,e1:sub_bfp16_e
ex10|ex10|5|62,74,185,197|x10:sub_bfp16_x,e10:sub_bfp16_e
ex11|ex11|5|63,75,186,198|x11:sub_bfp16_x,e11:sub_bfp16_e
ex2|ex2|1|64,76,187,199|x2:sub_bfp16_x,e2:sub_bfp16_e
ex3|ex3|1|65,77,188,200|x3:sub_bfp16_x,e3:sub_bfp16_e
ex4|ex4|2|66,78,189,201|x4:sub_bfp16_x,e4:sub_bfp16_e
ex5|ex5|2|67,79,190,202|x5:sub_bfp16_x,e5:sub_bfp16_e
ex6|ex6|3|68,80,191,203|x6:sub_bfp16_x,e6:sub_bfp16_e
ex7|ex7|3|69,81,192,204|x7:sub_bfp16_x,e7:sub_bfp16_e
ex8|ex8|4|70,82,193,205|x8:sub_bfp16_x,e8:sub_bfp16_e
ex9|ex9|4|71,83,194,206|x9:sub_bfp16_x,e9:sub_bfp16_e
ey0|ey0|0|60,61,72,73,183,184,195,196|ex0:sub_bfp576_lo,ex1:sub_bfp576_hi
ey1|ey1|1|64,65,76,77,187,188,199,200|ex2:sub_bfp576_lo,ex3:sub_bfp576_hi
ey2|ey2|2|66,67,78,79,189,190,201,202|ex4:sub_bfp576_lo,ex5:sub_bfp576_hi
ey3|ey3|3|68,69,80,81,191,192,203,204|ex6:sub_bfp576_lo,ex7:sub_bfp576_hi
ey4|ey4|4|70,71,82,83,193,194,205,206|ex8:sub_bfp576_lo,ex9:sub_bfp576_hi
ey5|ey5|5|62,63,74,75,185,186,197,198|ex10:sub_bfp576_lo,ex11:sub_bfp576_hi
l0|r1:r0|0|120,121|r0:sub_l_even,r1:sub_l_odd
l1|r3:r2|1|132,143|r2:sub_l_even,r3:sub_l_odd
l10|r21:r20|10|133,134|r20:sub_l_even,r21:sub_l_odd
l11|r23:r22|11|135,136|r22:sub_l_even,r23:sub_l_odd
l12|r25:r24|12|137,138|r24:sub_l_even,r25:sub_l_odd
l13|r27:r26|13|139,140|r26:sub_l_even,r27:sub_l_odd
l14|r29:r28|14|141,142|r28:sub_l_even,r29:sub_l_odd
l15|r31:r30|15|144,145|r30:sub_l_even,r31:sub_l_odd
l2|r5:r4|2|146,147|r4:sub_l_even,r5:sub_l_odd
l3|r7:r6|3|148,149|r6:sub_l_even,r7:sub_l_odd
l4|r9:r8|4|150,151|r8:sub_l_even,r9:sub_l_odd
l5|r11:r10|5|122,123|r10:sub_l_even,r11:sub_l_odd
l6|r13:r12|6|124,125|r12:sub_l_even,r13:sub_l_odd
l7|r15:r14|7|126,127|r14:sub_l_even,r15:sub_l_odd
l8|r17:r16|8|128,129|r16:sub_l_even,r17:sub_l_odd
l9|r19:r18|9|130,131|r18:sub_l_even,r19:sub_l_odd
lc|lc|55|84|
le|le|119|85|
lf0|lf0|0|87,89|lfl0:sub_lo_fifo,lfh0:sub_hi_fifo
lf1|lf1|1|88,90|lfl1:sub_lo_fifo,lfh1:sub_hi_fifo
lfe|lfe|2|86|
lfh0|lfh0|0|87|
lfh1|lfh1|1|88|
lfl0|lfl0|0|89|
lfl1|lfl1|1|90|
lr|lr|15|91|
ls|ls|79|92|
m0|m0|0|93|
m1|m1|1|94|
m2|m2|2|95|
m3|m3|3|96|
m4|m4|4|97|
m5|m5|5|98|
m6|m6|6|99|
m7|m7|7|100|
p0|p0|0|101|
p1|p1|1|102|
p2|p2|2|103|
p3|p3|3|104|
p4|p4|4|105|
p5|p5|5|106|
p6|p6|6|107|
p7|p7|7|108|
packSign0|packsign0|0|109|
packSign1|packsign1|1|110|
pe2_ads|pe2_ads|0|111|
plfr0|plfr0|0|87,89,101,137|p0:sub_ptr,lf0:sub_fifo,r24:sub_avail
plfr1|plfr1|1|88,90,102,138|p1:sub_ptr,lf1:sub_fifo,r25:sub_avail
q0|q0|0|112,116|ql0:sub_lo_mask,qh0:sub_hi_mask
q1|q1|1|113,117|ql1:sub_lo_mask,qh1:sub_hi_mask
q2|q2|2|114,118|ql2:sub_lo_mask,qh2:sub_hi_mask
q3|q3|3|115,119|ql3:sub_lo_mask,qh3:sub_hi_mask
qewh0|qewh0|0|60,112,183|qh0:sub_q_vecmaskexp_352,eh0:sub_e_vecmaskexp_352,wh0:sub_w_vecmaskexp_352
qewh1|qewh1|1|61,113,184|qh1:sub_q_vecmaskexp_352,eh1:sub_e_vecmaskexp_352,wh1:sub_w_vecmaskexp_352
qewh2|qewh2|2|64,114,187|qh2:sub_q_vecmaskexp_352,eh2:sub_e_vecmaskexp_352,wh2:sub_w_vecmaskexp_352
qewh3|qewh3|3|65,115,188|qh3:sub_q_vecmaskexp_352,eh3:sub_e_vecmaskexp_352,wh3:sub_w_vecmaskexp_352
qewl0|qewl0|0|72,116,195|ql0:sub_q_vecmaskexp_352,el0:sub_e_vecmaskexp_352,wl0:sub_w_vecmaskexp_352
qewl1|qewl1|1|73,117,196|ql1:sub_q_vecmaskexp_352,el1:sub_e_vecmaskexp_352,wl1:sub_w_vecmaskexp_352
qewl2|qewl2|2|76,118,199|ql2:sub_q_vecmaskexp_352,el2:sub_e_vecmaskexp_352,wl2:sub_w_vecmaskexp_352
qewl3|qewl3|3|77,119,200|ql3:sub_q_vecmaskexp_352,el3:sub_e_vecmaskexp_352,wl3:sub_w_vecmaskexp_352
qex0|qex0|0|60,72,112,116,183,195|qewl0:sub_lo_vecmaskexp_352,qewh0:sub_hi_vecmaskexp_352
qex1|qex1|0|61,73,113,117,184,196|qewl1:sub_lo_vecmaskexp_352,qewh1:sub_hi_vecmaskexp_352
qex2|qex2|1|64,76,114,118,187,199|qewl2:sub_lo_vecmaskexp_352,qewh2:sub_hi_vecmaskexp_352
qex3|qex3|1|65,77,115,119,188,200|qewl3:sub_lo_vecmaskexp_352,qewh3:sub_hi_vecmaskexp_352
qey0|qey0|0|60,61,72,73,112,113,116,117,183,184,195,196|qex0:sub_even_vecmaskexp_704,qex1:sub_odd_vecmaskexp_704
qey1|qey1|1|64,65,76,77,114,115,118,119,187,188,199,200|qex2:sub_even_vecmaskexp_704,qex3:sub_odd_vecmaskexp_704
qh0|qh0|0|112|
qh1|qh1|1|113|
qh2|qh2|2|114|
qh3|qh3|3|115|
ql0|ql0|0|116|
ql1|ql1|1|117|
ql2|ql2|2|118|
ql3|ql3|3|119|
qwh0|qwh0|0|112,183|qh0:sub_q,wh0:sub_w
qwh1|qwh1|1|113,184|qh1:sub_q,wh1:sub_w
qwh2|qwh2|2|114,187|qh2:sub_q,wh2:sub_w
qwh3|qwh3|3|115,188|qh3:sub_q,wh3:sub_w
qwl0|qwl0|0|116,195|ql0:sub_q,wl0:sub_w
qwl1|qwl1|1|117,196|ql1:sub_q,wl1:sub_w
qwl2|qwl2|2|118,199|ql2:sub_q,wl2:sub_w
qwl3|qwl3|3|119,200|ql3:sub_q,wl3:sub_w
qx0|qx0|0|112,116,183,195|x0:sub_sparse_x,q0:sub_sparse_q
qx1|qx1|0|113,117,184,196|x1:sub_sparse_x,q1:sub_sparse_q
qx2|qx2|1|114,118,187,199|x2:sub_sparse_x,q2:sub_sparse_q
qx3|qx3|1|115,119,188,200|x3:sub_sparse_x,q3:sub_sparse_q
qy0|qy0|0|112,113,116,117,183,184,195,196|qx0:sub_even_vecmask_640,qx1:sub_odd_vecmask_640
qy1|qy1|1|114,115,118,119,187,188,199,200|qx2:sub_even_vecmask_640,qx3:sub_odd_vecmask_640
r0|r0|0|120|
r1|r1|1|121|
r10|r10|10|122|
r11|r11|11|123|
r12|r12|12|124|
r13|r13|13|125|
r14|r14|14|126|
r15|r15|15|127|
r16|r16|16|128|
r17|r17|17|129|
r18|r18|18|130|
r19|r19|19|131|
r2|r2|2|132|
r20|r20|20|133|
r21|r21|21|134|
r22|r22|22|135|
r23|r23|23|136|
r24|r24|24|137|
r25|r25|25|138|
r26|r26|26|139|
r27|r27|27|140|
r28|r28|28|141|
r29|r29|29|142|
r3|r3|3|143|
r30|r30|30|144|
r31|r31|31|145|
r4|r4|4|146|
r5|r5|5|147|
r6|r6|6|148|
r7|r7|7|149|
r8|r8|8|150|
r9|r9|9|151|
s0|s0|0|152|
s1|s1|1|153|
s2|s2|2|154|
s3|s3|3|155|
sf|sf|1|156,157|sfl:sub_lo_fifo,sfh:sub_hi_fifo
sfh|sfh|6|156|
sfl|sfl|3|157|
sp|sp|47|158|
srCarry|srcarry|0|159|
srF2BFlags|srf2bflags|1|160|
srF2FFlags|srf2fflags|2|161|
srF2IFlags|srf2iflags|3|162|
srFPCnvFl2Fx|srfpcnvfl2fx|4|163|
srFPCnvFx2Fl|srfpcnvfx2fl|5|164|
srFPFlags|srfpflags|6|165|
srFPNlf|srfpnlf|7|166|
srFifo_of|srfifo_of|8|167|
srFifo_uf|srfifo_uf|9|168|
srMS0|srms0|10|169|
srSRS_of|srsrs_of|11|170|
srSS0|srss0|12|171|
srSparse_of|srsparse_of|13|172|
srUPS_of|srups_of|14|173|
srsSign0|srssign0|0|174|
srsSign1|srssign1|1|175|
tile_cntr|tile_cntr|0|176|
unpackSign0|unpacksign0|0|177|
unpackSign1|unpacksign1|1|178|
upsSign0|upssign0|0|179|
upsSign1|upssign1|1|180|
vaddSign0|vaddsign0|0|181|
vaddSign1|vaddsign1|1|182|
wh0|wh0|0|183|
wh1|wh1|0|184|
wh10|wh10|5|185|
wh11|wh11|5|186|
wh2|wh2|1|187|
wh3|wh3|1|188|
wh4|wh4|2|189|
wh5|wh5|2|190|
wh6|wh6|3|191|
wh7|wh7|3|192|
wh8|wh8|4|193|
wh9|wh9|4|194|
wl0|wl0|0|195|
wl1|wl1|0|196|
wl10|wl10|5|197|
wl11|wl11|5|198|
wl2|wl2|1|199|
wl3|wl3|1|200|
wl4|wl4|2|201|
wl5|wl5|2|202|
wl6|wl6|3|203|
wl7|wl7|3|204|
wl8|wl8|4|205|
wl9|wl9|4|206|
x0|x0|0|183,195|wl0:sub_256_lo,wh0:sub_256_hi
x1|x1|0|184,196|wl1:sub_256_lo,wh1:sub_256_hi
x10|x10|5|185,197|wl10:sub_256_lo,wh10:sub_256_hi
x11|x11|5|186,198|wl11:sub_256_lo,wh11:sub_256_hi
x2|x2|1|187,199|wl2:sub_256_lo,wh2:sub_256_hi
x3|x3|1|188,200|wl3:sub_256_lo,wh3:sub_256_hi
x4|x4|2|189,201|wl4:sub_256_lo,wh4:sub_256_hi
x5|x5|2|190,202|wl5:sub_256_lo,wh5:sub_256_hi
x6|x6|3|191,203|wl6:sub_256_lo,wh6:sub_256_hi
x7|x7|3|192,204|wl7:sub_256_lo,wh7:sub_256_hi
x8|x8|4|193,205|wl8:sub_256_lo,wh8:sub_256_hi
x9|x9|4|194,206|wl9:sub_256_lo,wh9:sub_256_hi
y0|y0|0|183,184,195,196|x0:sub_512_lo,x1:sub_512_hi
y1|y1|1|187,188,199,200|x2:sub_512_lo,x3:sub_512_hi
y2|y2|2|189,190,201,202|x4:sub_512_lo,x5:sub_512_hi
y3|y3|3|191,192,203,204|x6:sub_512_lo,x7:sub_512_hi
y4|y4|4|193,194,205,206|x8:sub_512_lo,x9:sub_512_hi
y5|y5|5|185,186,197,198|x10:sub_512_lo,x11:sub_512_hi
"""

_REGISTER_LAYOUT_RECORDS = (
    (20, 32, 32, 32),
    (32, 32, 32, 32),
    (64, 64, 64, 32),
    (80, 32, 128, 32),
    (128, 128, 128, 128),
    (160, 32, 256, 32),
    (256, 256, 256, 256),
    (512, 512, 512, 512),
    (576, 576, 576, 512),
    (640, 640, 640, 512),
    (704, 704, 704, 512),
    (1024, 1024, 1024, 512),
    (1088, 1088, 1088, 512),
    (1152, 1152, 1152, 512),
    (1280, 1280, 1280, 512),
    (1408, 1408, 1408, 512),
    (2048, 2048, 2048, 512),
)

_VALUE_TYPE_GROUPS = (
    ("i128", "v16i8", "v8i16", "v8bf16", "v4i32", "v4f32"),
    ("i20",),
    ("i32",),
    ("i64", "v8i8", "v4i16", "v4bf16", "v2i32", "v2f32", "v64i1"),
    ("v128i8", "v64i16", "v64bf16", "v32i32", "v16i64", "v32f32"),
    ("v16i32", "v8i64", "v16f32"),
    ("v32i32", "v16i64", "v32f32"),
    ("v32i8", "v16i16", "v16bf16", "v8f32", "v8i32"),
    ("v64i32", "v32i64", "v64f32"),
    ("v64i8", "v32i16", "v32bf16", "v16i32", "v8i64", "v16f32"),
    ("v8i1", "v16i1", "v32i1", "bf16", "i20", "v2bf16", "i32", "f32", "v4i8", "v2i16"),
    ("v8i8",),
)

_REGISTER_CLASS_RECORDS = """\
ACC1024|11|6|7|cml0,cml1,cml2,cml3,cml4,cmh0,cmh1,cmh2,cmh3,cmh4
ACC2048|16|8|7|dm0,dm1,dm2,dm3,dm4
ACC512|7|5|7|bmll0,bmll1,bmll2,bmll3,bmll4,bmlh0,bmlh1,bmlh2,bmlh3,bmlh4,bmhl0,bmhl1,bmhl2,bmhl3,bmhl4,bmhh0,bmhh1,bmhh2,bmhh3,bmhh4
EXPVEC64|2|11|7|e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11
FIFO1024|11|4|7|lf0,lf1,sf
FIFO512|7|9|7|sfh,sfl,lfh0,lfh1,lfl0,lfl1,lfe
SPARSEVEC1280|14|2|7|qy0,qy1
SPARSEVEC640|9|2|7|qx0,qx2,qx1,qx3
VEC1024|11|4|7|y0,y1,y2,y3,y4,y5
VEC128|4|0|7|q0,q2,q1,q3
VEC256|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10,wh1,wh3,wh5,wh7,wh9,wh11,wl0,wl2,wl4,wl6,wl8,wl10,wl1,wl3,wl5,wl7,wl9,wl11
VEC512|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
VEC576|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
eBMHH|7|5|7|bmhh0,bmhh1,bmhh2,bmhh3,bmhh4
eBMHL|7|5|7|bmhl0,bmhl1,bmhl2,bmhl3,bmhl4
eBMLH|7|5|7|bmlh0,bmlh1,bmlh2,bmlh3,bmlh4
eBMLL|7|5|7|bmll0,bmll1,bmll2,bmll3,bmll4
eBMSHH|7|5|7|bmhh0,bmhh1,bmhh2,bmhh3
eBMSHL|7|5|7|bmhl0,bmhl1,bmhl2,bmhl3
eBMSLH|7|5|7|bmlh0,bmlh1,bmlh2,bmlh3
eBMSLL|7|5|7|bmll0,bmll1,bmll2,bmll3
eCMH|11|6|7|cmh0,cmh1,cmh2,cmh3,cmh4
eCML|11|6|7|cml0,cml1,cml2,cml3,cml4
eD|3|1|7|d0,d1,d2,d3,d4,d5,d6,d7
eDC|0|1|7|dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7
eDCH|0|1|7|dc4,dc5,dc6,dc7
eDCL|0|1|7|dc0,dc1,dc2,dc3
eDC_as_32Bit|1|10|7|dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7
eDJ|0|1|7|dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7
eDJH|0|1|7|dj4,dj5,dj6,dj7
eDJL|0|1|7|dj0,dj1,dj2,dj3
eDJ_as_32Bit|1|10|7|dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7
eDM|16|8|7|dm0,dm1,dm2,dm3,dm4
eDN|0|1|7|dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7
eDNH|0|1|7|dn4,dn5,dn6,dn7
eDNL|0|1|7|dn0,dn1,dn2,dn3
eDN_as_32Bit|1|10|7|dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7
eDS|5|1|7|d0_3d,d1_3d,d2_3d,d3_3d
eE|1|10|7|el0,el2,el4,el6,el8,el10,el1,el3,el5,el7,el9,el11,eh0,eh2,eh4,eh6,eh8,eh10,eh1,eh3,eh5,eh7,eh9,eh11
eEXe|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10
eEXo|8|2|7|ex1,ex3,ex5,ex7,ex9,ex11
eEY|13|2|7|ey0,ey1,ey2,ey3,ey4,ey5
eEhe|1|10|7|eh0,eh2,eh4,eh6,eh8,eh10
eEho|1|10|7|eh1,eh3,eh5,eh7,eh9,eh11
eEle|1|10|7|el0,el2,el4,el6,el8,el10
eElo|1|10|7|el1,el3,el5,el7,el9,el11
eL|2|3|7|l0,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15
eLdFifoHReg|7|9|7|lfh0,lfh1
eLdFifoLReg|7|9|7|lfl0,lfl1
eLdFifoReg|11|4|7|lf0,lf1
eM|0|1|7|m0,m1,m2,m3,m4,m5,m6,m7
eM_as_32Bit|1|10|7|m0,m1,m2,m3,m4,m5,m6,m7
eP|0|1|7|p0,p1,p2,p3,p4,p5,p6,p7
ePS|0|1|7|p0,p1
ePSRFLdF|12|2|7|plfr0,plfr1
eP_as_32Bit|1|10|7|p0,p1,p2,p3,p4,p5,p6,p7
ePackSign|1|10|7|packSign0,packSign1
eQEXse|10|2|7|qex0,qex2
eQEXso|10|2|7|qex1,qex3
eQEYs|15|2|7|qey0,qey1
eQQse|4|0|7|q0,q2
eQQso|4|0|7|q1,q3
eQXs|9|2|7|qx0,qx2,qx1,qx3
eQXse|9|2|7|qx0,qx2
eQXsea|9|2|7|qx0,qx2
eQXseb|9|2|7|qx0,qx2
eQXsem|9|2|7|qx0,qx2
eQXses|9|2|7|qx0,qx2
eQXsew|9|2|7|qx0,qx2
eQXso|9|2|7|qx1,qx3
eQXsoa|9|2|7|qx1,qx3
eQXsob|9|2|7|qx1,qx3
eQXsom|9|2|7|qx1,qx3
eQXsos|9|2|7|qx1,qx3
eQXsow|9|2|7|qx1,qx3
eQYs|14|2|7|qy0,qy1
eR|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
eR16|1|10|7|r16
eR26|1|10|7|r26
eR27|1|10|7|r27
eR28|1|10|7|r28
eR29|1|10|7|r29
eR30|1|10|7|r30
eR31|1|10|7|r31
eRCR|1|10|7|crSat,crRnd,crFPMask,crF2IMask,crF2FMask,crF2BMask,crSRSMode,crUPSMode,crUnpackSize,crPackSize,srsSign0,srsSign1,upsSign0,upsSign1,packSign0,packSign1,unpackSign0,unpackSign1,vaddSign0,vaddSign1,crSCDEn,crMCDEn,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
eRF2|1|10|7|r24,r25
eRS16|1|10|7|r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
eRS4|1|10|7|r16,r17,r18,r19
eS|1|10|7|s0,s1,s2,s3
eSP|0|1|7|sp
eSP_as_32Bit|1|10|7|sp
eSRSSign|1|10|7|srsSign0,srsSign1
eSpecial20|0|1|7|ls,lr,le,sp,CORE_ID
eSpecial20_as_32Bit|1|10|7|ls,lr,le,sp,CORE_ID
eUPSSign|1|10|7|upsSign0,upsSign1
eUnpackSign|1|10|7|unpackSign0,unpackSign1
eVaddSign|1|10|7|vaddSign0,vaddSign1
eWH|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10,wh1,wh3,wh5,wh7,wh9,wh11
eWL|6|7|7|wl0,wl2,wl4,wl6,wl8,wl10,wl1,wl3,wl5,wl7,wl9,wl11
eWhe|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10
eWho|6|7|7|wh1,wh3,wh5,wh7,wh9,wh11
eWle|6|7|7|wl0,wl2,wl4,wl6,wl8,wl10
eWlo|6|7|7|wl1,wl3,wl5,wl7,wl9,wl11
eXe|7|9|7|x0,x2,x4,x6,x8,x10
eXo|7|9|7|x1,x3,x5,x7,x9,x11
eY|11|4|7|y0,y1,y2,y3,y4,y5
mAguDst|0|1|7|p0,p1,p2,p3,p4,p5,p6,p7,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7
mAguSrc|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7,p0,p1,p2,p3,p4,p5,p6,p7
mAluCg|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,lc
mBMHHm|7|5|7|bmhh0,bmhh1,bmhh2,bmhh3,bmhh4
mBMHLm|7|5|7|bmhl0,bmhl1,bmhl2,bmhl3,bmhl4
mBMLHm|7|5|7|bmlh0,bmlh1,bmlh2,bmlh3,bmlh4
mBMLLm|7|5|7|bmll0,bmll1,bmll2,bmll3,bmll4
mBMSm|7|5|7|bmll0,bmll1,bmll2,bmll3,bmlh0,bmlh1,bmlh2,bmlh3,bmhl0,bmhl1,bmhl2,bmhl3,bmhh0,bmhh1,bmhh2,bmhh3
mBMm|7|5|7|bmll0,bmll1,bmll2,bmll3,bmll4,bmlh0,bmlh1,bmlh2,bmlh3,bmlh4,bmhl0,bmhl1,bmhl2,bmhl3,bmhl4,bmhh0,bmhh1,bmhh2,bmhh3,bmhh4
mBMs|7|5|7|bmll0,bmll1,bmll2,bmll3,bmll4,bmlh0,bmlh1,bmlh2,bmlh3,bmlh4,bmhl0,bmhl1,bmhl2,bmhl3,bmhl4,bmhh0,bmhh1,bmhh2,bmhh3,bmhh4
mBp2Bp|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
mCMHm|11|6|7|cmh0,cmh1,cmh2,cmh3,cmh4
mCMHs|11|6|7|cmh0,cmh1,cmh2,cmh3,cmh4
mCMLm|11|6|7|cml0,cml1,cml2,cml3,cml4
mCMLs|11|6|7|cml0,cml1,cml2,cml3,cml4
mCMm|11|6|7|cml0,cml1,cml2,cml3,cml4,cmh0,cmh1,cmh2,cmh3,cmh4
mCMs|11|6|7|cml0,cml1,cml2,cml3,cml4,cmh0,cmh1,cmh2,cmh3,cmh4
mCRF2BMask|1|10|7|crF2BMask
mCRF2FMask|1|10|7|crF2FMask
mCRF2IMask|1|10|7|crF2IMask
mCRFP|1|2|7|crFPNlfMask,crFPCnvFx2FlMask,crFPCnvFl2FxMask
mCRFPCnvFl2FxMask|1|10|7|crFPCnvFl2FxMask
mCRFPCnvFx2FlMask|1|10|7|crFPCnvFx2FlMask
mCRFPMask|1|10|7|crFPMask
mCRFPNlfMask|1|10|7|crFPNlfMask
mCRMCDEn|1|10|7|crMCDEn
mCRPackSign|1|10|7|packSign0,packSign1
mCRPackSize|1|10|7|crPackSize
mCRRnd|1|10|7|crRnd
mCRSCDEn|1|10|7|crSCDEn
mCRSRSMode|1|10|7|crSRSMode
mCRSRSSign|1|10|7|srsSign0,srsSign1
mCRSat|1|10|7|crSat
mCRUPSMode|1|10|7|crUPSMode
mCRUPSSign|1|10|7|upsSign0,upsSign1
mCRUnpackSign|1|10|7|unpackSign0,unpackSign1
mCRUnpackSize|1|10|7|crUnpackSize
mCRVaddSign|1|10|7|vaddSign0,vaddSign1
mCRm|1|2|7|crSat,crRnd,crFPMask,crF2IMask,crF2FMask,crF2BMask,crSRSMode,crUPSMode,crUnpackSize,crPackSize,srsSign0,srsSign1,upsSign0,upsSign1,packSign0,packSign1,unpackSign0,unpackSign1,vaddSign0,vaddSign1,crSCDEn,crMCDEn
mCoreID|0|1|7|CORE_ID
mDCHa|0|1|7|dc4,dc5,dc6,dc7
mDCHb|0|1|7|dc4,dc5,dc6,dc7
mDCHs|0|1|7|dc4,dc5,dc6,dc7
mDCLa|0|1|7|dc0,dc1,dc2,dc3
mDCLb|0|1|7|dc0,dc1,dc2,dc3
mDCLs|0|1|7|dc0,dc1,dc2,dc3
mDCa|0|1|7|dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7
mDCb|0|1|7|dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7
mDCm|0|1|7|dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7
mDCs|0|1|7|dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7
mDJHa|0|1|7|dj4,dj5,dj6,dj7
mDJHb|0|1|7|dj4,dj5,dj6,dj7
mDJHs|0|1|7|dj4,dj5,dj6,dj7
mDJLa|0|1|7|dj0,dj1,dj2,dj3
mDJLb|0|1|7|dj0,dj1,dj2,dj3
mDJLs|0|1|7|dj0,dj1,dj2,dj3
mDJa|0|1|7|dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7
mDJb|0|1|7|dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7
mDJm|0|1|7|dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7
mDJs|0|1|7|dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7
mDMa|16|8|7|dm0,dm1,dm2,dm3,dm4
mDMm|16|8|7|dm0,dm1,dm2,dm3,dm4
mDMs|16|8|7|dm0,dm1,dm2,dm3,dm4
mDNHa|0|1|7|dn4,dn5,dn6,dn7
mDNHb|0|1|7|dn4,dn5,dn6,dn7
mDNHs|0|1|7|dn4,dn5,dn6,dn7
mDNLa|0|1|7|dn0,dn1,dn2,dn3
mDNLb|0|1|7|dn0,dn1,dn2,dn3
mDNLs|0|1|7|dn0,dn1,dn2,dn3
mDNa|0|1|7|dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7
mDNb|0|1|7|dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7
mDNm|0|1|7|dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7
mDNs|0|1|7|dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7
mDSa|5|1|7|d0_3d,d1_3d,d2_3d,d3_3d
mDSb|5|1|7|d0_3d,d1_3d,d2_3d,d3_3d
mDSs|5|1|7|d0_3d,d1_3d,d2_3d,d3_3d
mDa|3|1|7|d0,d1,d2,d3,d4,d5,d6,d7
mDb|3|1|7|d0,d1,d2,d3,d4,d5,d6,d7
mDm|0|1|7|dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7
mDs|3|1|7|d0,d1,d2,d3,d4,d5,d6,d7
mEXa|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
mEXb|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
mEXea|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10
mEXm|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
mEXn|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
mEXoa|8|2|7|ex1,ex3,ex5,ex7,ex9,ex11
mEXs|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
mEXv|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
mEXw|8|2|7|ex0,ex2,ex4,ex6,ex8,ex10,ex1,ex3,ex5,ex7,ex9,ex11
mEYv|13|2|7|ey0,ey1,ey2,ey3,ey4,ey5
mEYw|13|2|7|ey0,ey1,ey2,ey3,ey4,ey5
mEhea|1|10|7|eh0,eh2,eh4,eh6,eh8,eh10
mEhm|1|10|7|eh0,eh2,eh4,eh6,eh8,eh10,eh1,eh3,eh5,eh7,eh9,eh11
mEhoa|1|10|7|eh1,eh3,eh5,eh7,eh9,eh11
mElm|1|10|7|el0,el2,el4,el6,el8,el10,el1,el3,el5,el7,el9,el11
mEs|1|10|7|el0,el2,el4,el6,el8,el10,el1,el3,el5,el7,el9,el11,eh0,eh2,eh4,eh6,eh8,eh10,eh1,eh3,eh5,eh7,eh9,eh11
mFifoExtra|7|9|7|lfe
mFifoHLReg|7|9|7|sfh,sfl,lfh0,lfh1,lfl0,lfl1,lfe
mFl2FxSrc_W|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10,wh1,wh3,wh5,wh7,wh9,wh11,wl0,wl2,wl4,wl6,wl8,wl10,wl1,wl3,wl5,wl7,wl9,wl11
mFlt2fx|1|10|7|s3
mFp2B0|16|8|7|dm0,dm1,dm2,dm3,dm4
mFp2B1|16|8|7|dm0,dm1,dm2,dm3,dm4
mFx2flt|1|10|7|s2
mL8m|2|3|7|l8
mLCaCg|1|10|7|lc
mLCm|1|10|7|lc
mLCmCg|1|10|7|lc
mLEm|1|10|7|le
mLRa|0|1|7|lr
mLRm|1|10|7|lr
mLRs|0|1|7|lr
mLRx|0|1|7|lr
mLSm|1|10|7|ls
mLdFifo_a|7|9|7|lfl0,lfh0,lfl1,lfh1
mLdFifo_b|7|9|7|lfl0,lfh0,lfl1,lfh1
mLdaCg|1|10|7|p0,p1,p2,p3,p4,p5,p6,p7,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7,lc
mLdaScl|1|10|7|p0,p1,p2,p3,p4,p5,p6,p7,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7,el0,el2,el4,el6,el8,el10,el1,el3,el5,el7,el9,el11,eh0,eh2,eh4,eh6,eh8,eh10,eh1,eh3,eh5,eh7,eh9,eh11,lr
mLm|2|3|7|l0,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15
mLockId_reg|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mMStream_tlast_reg|1|10|7|r28
mMa|0|1|7|m0,m1,m2,m3,m4,m5,m6,m7
mMb|0|1|7|m0,m1,m2,m3,m4,m5,m6,m7
mMcdBMSrc|7|5|7|bmll0,bmll1,bmll2,bmll3,bmll4,bmlh0,bmlh1,bmlh2,bmlh3,bmlh4,bmhl0,bmhl1,bmhl2,bmhl3,bmhl4,bmhh0,bmhh1,bmhh2,bmhh3,bmhh4
mMcdXSrc|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mMm|0|1|7|m0,m1,m2,m3,m4,m5,m6,m7
mMs|0|1|7|m0,m1,m2,m3,m4,m5,m6,m7
mMvBMXDst|7|9|2|bmll0,bmll1,bmll2,bmll3,bmll4,bmlh0,bmlh1,bmlh2,bmlh3,bmlh4,bmhl0,bmhl1,bmhl2,bmhl3,bmhl4,bmhh0,bmhh1,bmhh2,bmhh3,bmhh4,x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11,sfh,sfl,lfh0,lfh1,lfl0,lfl1,lfe
mMvBMXSrc|7|9|2|bmll0,bmll1,bmll2,bmll3,bmll4,bmlh0,bmlh1,bmlh2,bmlh3,bmlh4,bmhl0,bmhl1,bmhl2,bmhl3,bmhl4,bmhh0,bmhh1,bmhh2,bmhh3,bmhh4,x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11,sfh,sfl,lfh0,lfh1,lfl0,lfl1,lfe
mMvSclDst|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7,p0,p1,p2,p3,p4,p5,p6,p7,s0,s1,s2,s3,le,ls,lr,sp,crSat,crRnd,crFPMask,crF2IMask,crF2FMask,crF2BMask,crSRSMode,crUPSMode,crUnpackSize,crPackSize,srsSign0,srsSign1,upsSign0,upsSign1,packSign0,packSign1,unpackSign0,unpackSign1,vaddSign0,vaddSign1,crSCDEn,crMCDEn,srCarry,srSS0,srMS0,srSRS_of,srUPS_of,srFifo_of,srFifo_uf,srSparse_of,srFPFlags,srF2IFlags,srF2FFlags,srF2BFlags,srFPNlf,srFPCnvFx2Fl,srFPCnvFl2Fx,crFPNlfMask,crFPCnvFx2FlMask,crFPCnvFl2FxMask,lc
mMvSclDstCg|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7,p0,p1,p2,p3,p4,p5,p6,p7,s0,s1,s2,s3,le,ls,lr,sp,crSat,crRnd,crFPMask,crF2IMask,crF2FMask,crF2BMask,crSRSMode,crUPSMode,crUnpackSize,crPackSize,srsSign0,srsSign1,upsSign0,upsSign1,packSign0,packSign1,unpackSign0,unpackSign1,vaddSign0,vaddSign1,crSCDEn,crMCDEn,srCarry,srSS0,srMS0,srSRS_of,srUPS_of,srFifo_of,srFifo_uf,srSparse_of,srFPFlags,srF2IFlags,srF2FFlags,srF2BFlags,srFPNlf,srFPCnvFx2Fl,srFPCnvFl2Fx,crFPNlfMask,crFPCnvFx2FlMask,crFPCnvFl2FxMask,lc
mMvSclSrc|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7,p0,p1,p2,p3,p4,p5,p6,p7,s0,s1,s2,s3,le,ls,lr,sp,crSat,crRnd,crFPMask,crF2IMask,crF2FMask,crF2BMask,crSRSMode,crUPSMode,crUnpackSize,crPackSize,srsSign0,srsSign1,upsSign0,upsSign1,packSign0,packSign1,unpackSign0,unpackSign1,vaddSign0,vaddSign1,crSCDEn,crMCDEn,srCarry,srSS0,srMS0,srSRS_of,srUPS_of,srFifo_of,srFifo_uf,srSparse_of,srFPFlags,srF2IFlags,srF2FFlags,srF2BFlags,srFPNlf,srFPCnvFx2Fl,srFPCnvFl2Fx,crFPNlfMask,crFPCnvFx2FlMask,crFPCnvFl2FxMask,lc,CORE_ID
mOptConv|1|10|7|s2
mOptConvDel|1|10|7|s3
mPa|0|1|7|p0,p1,p2,p3,p4,p5,p6,p7
mPb|0|1|7|p0,p1,p2,p3,p4,p5,p6,p7
mPfa|0|1|7|p0,p1
mPfb|0|1|7|p0,p1
mPfs|0|1|7|p2
mPm|0|1|7|p0,p1,p2,p3,p4,p5,p6,p7
mPs|0|1|7|p0,p1,p2,p3,p4,p5,p6,p7
mQEXsa|10|2|7|qex0,qex2,qex1,qex3
mQEXsb|10|2|7|qex0,qex2,qex1,qex3
mQEXsm|10|2|7|qex0,qex2,qex1,qex3
mQEXss|10|2|7|qex0,qex2,qex1,qex3
mQEXsw|10|2|7|qex0,qex2,qex1,qex3
mQEYsw|15|2|7|qey0,qey1
mQQsa|4|0|7|q0,q2,q1,q3
mQQsm|4|0|7|q0,q2,q1,q3
mQQss|4|0|7|q0,q2,q1,q3
mQXsa|9|2|7|qx0,qx2,qx1,qx3
mQXsb|9|2|7|qx0,qx2,qx1,qx3
mQXsea|9|2|7|qx0,qx2
mQXsm|9|2|7|qx0,qx2,qx1,qx3
mQXsoa|9|2|7|qx1,qx3
mQXss|9|2|7|qx0,qx2,qx1,qx3
mQXsw|9|2|7|qx0,qx2,qx1,qx3
mQYsa|14|2|7|qy0,qy1
mQYsb|14|2|7|qy0,qy1
mQYsm|14|2|7|qy0,qy1
mQYss|14|2|7|qy0,qy1
mQYsw|14|2|7|qy0,qy1
mR16_vcompare|1|10|7|r16
mR26_fifo_st|1|10|7|r26
mR26_lock|1|10|7|r26
mR27_select|1|10|7|r27
mR28_tlast|1|10|7|r28
mR29_insert|1|10|7|r29
mR30_fifo_step_e1|1|10|7|r30
mR30_fifo_step_e7|1|10|7|r30
mR30_shiftx|1|10|7|r30
mR31_divs|1|10|7|r31
mR31_scd|1|10|7|r31
mRF2a|1|10|7|r24,r25
mRF2b|1|10|7|r24,r25
mRS16m|1|10|7|r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mRa|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mRm|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mRs|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mRv|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mRx|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mRy|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mRz|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
mS2|1|10|7|s2
mS3|1|10|7|s3
mSCD_r|1|10|7|r31
mSCD_r_incr|1|10|7|r31
mSPa|0|1|7|sp
mSPb|0|1|7|sp
mSPm|1|10|7|sp
mSPs|0|1|7|sp
mSRCarry|1|10|7|srCarry
mSRF2BFlags|1|10|7|srF2BFlags
mSRF2FFlags|1|10|7|srF2FFlags
mSRF2IFlags|1|10|7|srF2IFlags
mSRFPCnvFl2Fx|1|10|7|srFPCnvFl2Fx
mSRFPCnvFx2Fl|1|10|7|srFPCnvFx2Fl
mSRFPFlags|1|10|7|srFPFlags
mSRFPNlf|1|10|7|srFPNlf
mSRFifo_of|1|10|7|srFifo_of
mSRFifo_uf|1|10|7|srFifo_uf
mSRMS0|1|10|7|srMS0
mSRSRS_of|1|10|7|srSRS_of
mSRSS0|1|10|7|srSS0
mSRSparse_of|1|10|7|srSparse_of
mSRUPS_of|1|10|7|srUPS_of
mSRm|1|2|7|srCarry,srSS0,srMS0,srSRS_of,srUPS_of,srFifo_of,srFifo_uf,srSparse_of,srFPFlags,srF2IFlags,srF2FFlags,srF2BFlags,srFPNlf,srFPCnvFx2Fl,srFPCnvFl2Fx
mSa|1|10|7|s0,s1,s2,s3
mSclMS|1|10|7|r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7,p0,p1,p2,p3,p4,p5,p6,p7
mSclSt|1|10|7|p0,p1,p2,p3,p4,p5,p6,p7,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,m0,m1,m2,m3,m4,m5,m6,m7,el0,el2,el4,el6,el8,el10,el1,el3,el5,el7,el9,el11,eh0,eh2,eh4,eh6,eh8,eh10,eh1,eh3,eh5,eh7,eh9,eh11,lr
mShflBMDst|7|5|7|bmll0,bmll1,bmll2,bmll3,bmlh0,bmlh1,bmlh2,bmlh3,bmhl0,bmhl1,bmhl2,bmhl3,bmhh0,bmhh1,bmhh2,bmhh3
mShflXDst|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mSm|1|10|7|s0,s1,s2,s3
mSs|1|10|7|s0,s1,s2,s3
mStFifo|11|4|7|sf
mStFifoh|7|9|7|sfh
mStFifol|7|9|7|sfl
mWa|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10,wh1,wh3,wh5,wh7,wh9,wh11,wl0,wl2,wl4,wl6,wl8,wl10,wl1,wl3,wl5,wl7,wl9,wl11
mWb|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10,wh1,wh3,wh5,wh7,wh9,wh11,wl0,wl2,wl4,wl6,wl8,wl10,wl1,wl3,wl5,wl7,wl9,wl11
mWbfsrs|7|5|7|bmll0,bmll1,bmll2,bmll3,bmll4,bmlh0,bmlh1,bmlh2,bmlh3,bmlh4,bmhl0,bmhl1,bmhl2,bmhl3,bmhl4,bmhh0,bmhh1,bmhh2,bmhh3,bmhh4
mWhea|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10
mWheb|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10
mWhem|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10
mWhes|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10
mWhoa|6|7|7|wh1,wh3,wh5,wh7,wh9,wh11
mWhob|6|7|7|wh1,wh3,wh5,wh7,wh9,wh11
mWhom|6|7|7|wh1,wh3,wh5,wh7,wh9,wh11
mWhos|6|7|7|wh1,wh3,wh5,wh7,wh9,wh11
mWlea|6|7|7|wl0,wl2,wl4,wl6,wl8,wl10
mWleb|6|7|7|wl0,wl2,wl4,wl6,wl8,wl10
mWlem|6|7|7|wl0,wl2,wl4,wl6,wl8,wl10
mWles|6|7|7|wl0,wl2,wl4,wl6,wl8,wl10
mWloa|6|7|7|wl1,wl3,wl5,wl7,wl9,wl11
mWlob|6|7|7|wl1,wl3,wl5,wl7,wl9,wl11
mWlom|6|7|7|wl1,wl3,wl5,wl7,wl9,wl11
mWlos|6|7|7|wl1,wl3,wl5,wl7,wl9,wl11
mWm|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10,wh1,wh3,wh5,wh7,wh9,wh11,wl0,wl2,wl4,wl6,wl8,wl10,wl1,wl3,wl5,wl7,wl9,wl11
mWs|6|7|7|wh0,wh2,wh4,wh6,wh8,wh10,wh1,wh3,wh5,wh7,wh9,wh11,wl0,wl2,wl4,wl6,wl8,wl10,wl1,wl3,wl5,wl7,wl9,wl11
mXa|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mXb|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mXbfsrs|11|6|7|cml0,cml1,cml2,cml3,cml4,cmh0,cmh1,cmh2,cmh3,cmh4
mXea|7|9|7|x0,x2,x4,x6,x8,x10
mXem|7|9|7|x0,x2,x4,x6,x8,x10
mXen|7|9|7|x0,x2,x4,x6,x8,x10
mXm|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mXn|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mXoa|7|9|7|x1,x3,x5,x7,x9,x11
mXom|7|9|7|x1,x3,x5,x7,x9,x11
mXon|7|9|7|x1,x3,x5,x7,x9,x11
mXs|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mXv|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mXw|7|9|7|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11
mYb|11|4|7|y0,y1,y2,y3,y4,y5
mYs|11|4|7|y0,y1,y2,y3,y4,y5
mYv|11|4|7|y0,y1,y2,y3,y4,y5
mYw|11|4|7|y0,y1,y2,y3,y4,y5
spill_eDC_to_eR|1|10|7|dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
spill_eDJ_to_eR|1|10|7|dj0,dj1,dj2,dj3,dj4,dj5,dj6,dj7,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7
spill_eDN_to_eR|1|10|7|dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
spill_eM_to_eR|1|10|7|m0,m1,m2,m3,m4,m5,m6,m7,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,dn0,dn1,dn2,dn3,dn4,dn5,dn6,dn7
spill_eP_to_eR|1|10|7|p0,p1,p2,p3,p4,p5,p6,p7,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
spill_eS_to_eR|1|10|7|s0,s1,s2,s3,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31
spill_vec512_to_composite|7|9|5|x0,x2,x4,x6,x8,x10,x1,x3,x5,x7,x9,x11,bmll0,bmll1,bmll2,bmll3,bmll4,bmlh0,bmlh1,bmlh2,bmlh3,bmlh4,bmhl0,bmhl1,bmhl2,bmhl3,bmhl4,bmhh0,bmhh1,bmhh2,bmhh3,bmhh4,lfh0,lfh1,lfl0,lfl1,sfl,sfh,lfe
"""

_REGISTER_ENCODING_MAP_RECORDS = (
    "CORE_ID:87,crF2BMask:51,crF2FMask:115,crF2IMask:11,crFPCnvFl2FxMask:75,crFPCnvFx2FlMask:43,crFPMask:107,crFPNlfMask:27,crMCDEn:91,crPackSize:59,crRnd:123,crSCDEn:7,crSRSMode:71,crSat:39,crUPSMode:103,crUnpackSize:23,dc0:12,dc1:28,dc2:44,dc3:60,dc4:76,dc5:92,dc6:108,dc7:124,dj0:8,dj1:24,dj2:40,dj3:56,dj4:72,dj5:88,dj6:104,dj7:120,dn0:4,dn1:20,dn2:36,dn3:52,dn4:68,dn5:84,dn6:100,dn7:116,lc:55,le:119,lr:15,ls:79,m0:0,m1:16,m2:32,m3:48,m4:64,m5:80,m6:96,m7:112,p0:6,p1:22,p2:38,p3:54,p4:70,p5:86,p6:102,p7:118,packSign0:30,packSign1:94,r0:1,r1:5,r10:41,r11:45,r12:49,r13:53,r14:57,r15:61,r16:65,r17:69,r18:73,r19:77,r2:9,r20:81,r21:85,r22:89,r23:93,r24:97,r25:101,r26:105,r27:109,r28:113,r29:117,r3:13,r30:121,r31:125,r4:17,r5:21,r6:25,r7:29,r8:33,r9:37,s0:14,s1:46,s2:78,s3:110,sp:47,srCarry:2,srF2BFlags:10,srF2FFlags:18,srF2IFlags:26,srFPCnvFl2Fx:34,srFPCnvFx2Fl:42,srFPFlags:50,srFPNlf:58,srFifo_of:66,srFifo_uf:74,srMS0:82,srSRS_of:90,srSS0:98,srSparse_of:106,srUPS_of:114,srsSign0:62,srsSign1:126,unpackSign0:35,unpackSign1:99,upsSign0:3,upsSign1:67,vaddSign0:19,vaddSign1:83",
    "bmhh0:3,bmhh1:7,bmhh2:11,bmhh3:15,bmhh4:19,bmhl0:2,bmhl1:6,bmhl2:10,bmhl3:14,bmhl4:18,bmlh0:1,bmlh1:5,bmlh2:9,bmlh3:13,bmlh4:17,bmll0:0,bmll1:4,bmll2:8,bmll3:12,bmll4:16",
    "bmhh0:3,bmhh1:7,bmhh2:11,bmhh3:15,bmhl0:2,bmhl1:6,bmhl2:10,bmhl3:14,bmlh0:1,bmlh1:5,bmlh2:9,bmlh3:13,bmll0:0,bmll1:4,bmll2:8,bmll3:12",
    "bmhh0:6,bmhh1:14,bmhh2:22,bmhh3:30,bmhh4:38,bmhl0:4,bmhl1:12,bmhl2:20,bmhl3:28,bmhl4:36,bmlh0:2,bmlh1:10,bmlh2:18,bmlh3:26,bmlh4:34,bmll0:0,bmll1:8,bmll2:16,bmll3:24,bmll4:32,lfe:19,lfh0:3,lfh1:35,lfl0:11,lfl1:43,sfh:51,sfl:27,x0:1,x1:5,x10:41,x11:45,x2:9,x3:13,x4:17,x5:21,x6:25,x7:29,x8:33,x9:37",
    "cmh0:1,cmh1:3,cmh2:5,cmh3:7,cmh4:9,cml0:0,cml1:2,cml2:4,cml3:6,cml4:8",
    "crF2BMask:0,crF2FMask:2,crF2IMask:4,crFPMask:6,crMCDEn:11,crPackSize:8,crRnd:10,crSCDEn:27,crSRSMode:12,crSat:14,crUPSMode:16,crUnpackSize:18,packSign0:1,packSign1:17,srsSign0:3,srsSign1:19,unpackSign0:7,unpackSign1:23,upsSign0:5,upsSign1:21,vaddSign0:9,vaddSign1:25",
    "crF2BMask:51,crF2FMask:115,crF2IMask:11,crFPCnvFl2FxMask:75,crFPCnvFx2FlMask:43,crFPMask:107,crFPNlfMask:27,crMCDEn:91,crPackSize:59,crRnd:123,crSCDEn:7,crSRSMode:71,crSat:39,crUPSMode:103,crUnpackSize:23,dc0:12,dc1:28,dc2:44,dc3:60,dc4:76,dc5:92,dc6:108,dc7:124,dj0:8,dj1:24,dj2:40,dj3:56,dj4:72,dj5:88,dj6:104,dj7:120,dn0:4,dn1:20,dn2:36,dn3:52,dn4:68,dn5:84,dn6:100,dn7:116,lc:87,le:55,lr:119,ls:15,m0:0,m1:16,m2:32,m3:48,m4:64,m5:80,m6:96,m7:112,p0:6,p1:22,p2:38,p3:54,p4:70,p5:86,p6:102,p7:118,packSign0:30,packSign1:94,r0:1,r1:5,r10:41,r11:45,r12:49,r13:53,r14:57,r15:61,r16:65,r17:69,r18:73,r19:77,r2:9,r20:81,r21:85,r22:89,r23:93,r24:97,r25:101,r26:105,r27:109,r28:113,r29:117,r3:13,r30:121,r31:125,r4:17,r5:21,r6:25,r7:29,r8:33,r9:37,s0:14,s1:46,s2:78,s3:110,sp:79,srCarry:2,srF2BFlags:10,srF2FFlags:18,srF2IFlags:26,srFPCnvFl2Fx:34,srFPCnvFx2Fl:42,srFPFlags:50,srFPNlf:58,srFifo_of:66,srFifo_uf:74,srMS0:82,srSRS_of:90,srSS0:98,srSparse_of:106,srUPS_of:114,srsSign0:62,srsSign1:126,unpackSign0:35,unpackSign1:99,upsSign0:3,upsSign1:67,vaddSign0:19,vaddSign1:83",
    "dc0:0,dc1:8,dc2:16,dc3:24,dc4:32,dc5:40,dc6:48,dc7:56,dj0:1,dj1:9,dj2:17,dj3:25,dj4:33,dj5:41,dj6:49,dj7:57,dn0:2,dn1:10,dn2:18,dn3:26,dn4:34,dn5:42,dn6:50,dn7:58,m0:3,m1:11,m2:19,m3:27,m4:35,m5:43,m6:51,m7:59,p0:4,p1:12,p2:20,p3:28,p4:36,p5:44,p6:52,p7:60",
    "dc0:1,dc1:17,dc2:33,dc3:49,dc4:65,dc5:81,dc6:97,dc7:113,dj0:5,dj1:21,dj2:37,dj3:53,dj4:69,dj5:85,dj6:101,dj7:117,dn0:9,dn1:25,dn2:41,dn3:57,dn4:73,dn5:89,dn6:105,dn7:121,m0:13,m1:29,m2:45,m3:61,m4:77,m5:93,m6:109,m7:125,p0:2,p1:18,p2:34,p3:50,p4:66,p5:82,p6:98,p7:114,r0:0,r1:4,r10:40,r11:44,r12:48,r13:52,r14:56,r15:60,r16:64,r17:68,r18:72,r19:76,r2:8,r20:80,r21:84,r22:88,r23:92,r24:96,r25:100,r26:104,r27:108,r28:112,r29:116,r3:12,r30:120,r31:124,r4:16,r5:20,r6:24,r7:28,r8:32,r9:36",
    "dc0:3,dc1:7,dc2:11,dc3:15,dc4:19,dc5:23,dc6:27,dc7:31,dj0:2,dj1:6,dj2:10,dj3:14,dj4:18,dj5:22,dj6:26,dj7:30,dn0:1,dn1:5,dn2:9,dn3:13,dn4:17,dn5:21,dn6:25,dn7:29,m0:0,m1:4,m2:8,m3:12,m4:16,m5:20,m6:24,m7:28",
    "dc0:12,dc1:28,dc2:44,dc3:60,dc4:76,dc5:92,dc6:108,dc7:124,dj0:8,dj1:24,dj2:40,dj3:56,dj4:72,dj5:88,dj6:104,dj7:120,dn0:4,dn1:20,dn2:36,dn3:52,dn4:68,dn5:84,dn6:100,dn7:116,eh0:1,eh1:9,eh10:81,eh11:89,eh2:17,eh3:25,eh4:33,eh5:41,eh6:49,eh7:57,eh8:65,eh9:73,el0:5,el1:13,el10:85,el11:93,el2:21,el3:29,el4:37,el5:45,el6:53,el7:61,el8:69,el9:77,lr:7,m0:0,m1:16,m2:32,m3:48,m4:64,m5:80,m6:96,m7:112,p0:3,p1:19,p2:35,p3:51,p4:67,p5:83,p6:99,p7:115,r0:2,r1:6,r10:42,r11:46,r12:50,r13:54,r14:58,r15:62,r16:66,r17:70,r18:74,r19:78,r2:10,r20:82,r21:86,r22:90,r23:94,r24:98,r25:102,r26:106,r27:110,r28:114,r29:118,r3:14,r30:122,r31:126,r4:18,r5:22,r6:26,r7:30,r8:34,r9:38",
    "dc0:14,dc1:30,dc2:46,dc3:62,dc4:78,dc5:94,dc6:110,dc7:126,dj0:10,dj1:26,dj2:42,dj3:58,dj4:74,dj5:90,dj6:106,dj7:122,dn0:6,dn1:22,dn2:38,dn3:54,dn4:70,dn5:86,dn6:102,dn7:118,lc:21,m0:2,m1:18,m2:34,m3:50,m4:66,m5:82,m6:98,m7:114,p0:13,p1:29,p2:45,p3:61,p4:77,p5:93,p6:109,p7:125,r0:0,r1:4,r10:40,r11:44,r12:48,r13:52,r14:56,r15:60,r16:64,r17:68,r18:72,r19:76,r2:8,r20:80,r21:84,r22:88,r23:92,r24:96,r25:100,r26:104,r27:108,r28:112,r29:116,r3:12,r30:120,r31:124,r4:16,r5:20,r6:24,r7:28,r8:32,r9:36",
    "dc0:14,dc1:30,dc2:46,dc3:62,dc4:78,dc5:94,dc6:110,dc7:126,dj0:10,dj1:26,dj2:42,dj3:58,dj4:74,dj5:90,dj6:106,dj7:122,dn0:6,dn1:22,dn2:38,dn3:54,dn4:70,dn5:86,dn6:102,dn7:118,m0:2,m1:18,m2:34,m3:50,m4:66,m5:82,m6:98,m7:114,p0:13,p1:29,p2:45,p3:61,p4:77,p5:93,p6:109,p7:125,r0:0,r1:4,r10:40,r11:44,r12:48,r13:52,r14:56,r15:60,r16:64,r17:68,r18:72,r19:76,r2:8,r20:80,r21:84,r22:88,r23:92,r24:96,r25:100,r26:104,r27:108,r28:112,r29:116,r3:12,r30:120,r31:124,r4:16,r5:20,r6:24,r7:28,r8:32,r9:36",
    "eh0:0,eh1:1,eh10:10,eh11:11,eh2:2,eh3:3,eh4:4,eh5:5,eh6:6,eh7:7,eh8:8,eh9:9",
    "eh0:0,eh1:2,eh10:20,eh11:22,eh2:4,eh3:6,eh4:8,eh5:10,eh6:12,eh7:14,eh8:16,eh9:18,el0:1,el1:3,el10:21,el11:23,el2:5,el3:7,el4:9,el5:11,el6:13,el7:15,el8:17,el9:19",
    "el0:0,el1:1,el10:10,el11:11,el2:2,el3:3,el4:4,el5:5,el6:6,el7:7,el8:8,el9:9",
    "ex0:0,ex1:1,ex10:10,ex11:11,ex2:2,ex3:3,ex4:4,ex5:5,ex6:6,ex7:7,ex8:8,ex9:9",
    "lc:1,r0:0,r1:2,r10:20,r11:22,r12:24,r13:26,r14:28,r15:30,r16:32,r17:34,r18:36,r19:38,r2:4,r20:40,r21:42,r22:44,r23:46,r24:48,r25:50,r26:52,r27:54,r28:56,r29:58,r3:6,r30:60,r31:62,r4:8,r5:10,r6:12,r7:14,r8:16,r9:18",
    "lfe:2,lfh0:0,lfh1:4,lfl0:1,lfl1:5,sfh:6,sfl:3",
    "q0:0,q1:1,q2:2,q3:3",
    "q0:0,q1:3,q2:4,q3:7",
    "qex0:0,qex1:1,qex2:2,qex3:3",
    "qx0:0,qx1:1,qx2:2,qx3:3",
    "srCarry:0,srF2BFlags:1,srF2FFlags:2,srF2IFlags:3,srFPCnvFl2Fx:4,srFPCnvFx2Fl:5,srFPFlags:6,srFPNlf:7,srFifo_of:8,srFifo_uf:9,srMS0:10,srSRS_of:11,srSS0:12,srSparse_of:13,srUPS_of:14",
    "wh0:0,wh1:2,wh10:20,wh11:22,wh2:4,wh3:6,wh4:8,wh5:10,wh6:12,wh7:14,wh8:16,wh9:18,wl0:1,wl1:3,wl10:21,wl11:23,wl2:5,wl3:7,wl4:9,wl5:11,wl6:13,wl7:15,wl8:17,wl9:19",
    "x0:0,x1:1,x10:10,x11:11,x2:2,x3:3,x4:4,x5:5,x6:6,x7:7,x8:8,x9:9",
)

_REGISTER_ADAPTER_RECORDS = """\
OP_mAguDst|mAguDst|7|
OP_mAguSrc|mAguSrc|8|
OP_mAluCg|mAluCg|17|
OP_mBMSm|mBMSm|2|
OP_mBMm|mBMm|1|
OP_mBMs|mBMs|1|
OP_mCMm|mCMm|4|
OP_mCMs|mCMs|4|
OP_mCRm|mCRm|5|
OP_mDm|mDm|9|
OP_mEXa|mEXa|16|
OP_mEXb|mEXb|16|
OP_mEXm|mEXm|16|
OP_mEXn|mEXn|16|
OP_mEXs|mEXs|16|
OP_mEXv|mEXv|16|
OP_mEXw|mEXw|16|
OP_mEhm|mEhm|13|
OP_mElm|mElm|15|
OP_mEs|mEs|14|
OP_mFifoHLReg|mFifoHLReg|18|
OP_mFl2FxSrc_W|mFl2FxSrc_W|24|
OP_mLdaCg|mLdaCg|11|
OP_mLdaScl|mLdaScl|10|
OP_mMcdBMSrc|mMcdBMSrc|1|
OP_mMcdXSrc|mMcdXSrc|25|
OP_mMvBMXDst|mMvBMXDst|3|
OP_mMvBMXSrc|mMvBMXSrc|3|
OP_mMvSclDst|mMvSclDst|6|
OP_mMvSclDstCg|mMvSclDstCg|6|
OP_mMvSclSrc|mMvSclSrc|0|
OP_mQEXsa|mQEXsa|21|
OP_mQEXsb|mQEXsb|21|
OP_mQEXsm|mQEXsm|21|
OP_mQEXsw|mQEXsw|21|
OP_mQQsa|mQQsa|20|19
OP_mQQsm|mQQsm|20|19
OP_mQQss|mQQss|20|19
OP_mQXsa|mQXsa|22|
OP_mQXsb|mQXsb|22|
OP_mQXsm|mQXsm|22|
OP_mQXsw|mQXsw|22|
OP_mSRm|mSRm|23|
OP_mSclMS|mSclMS|12|
OP_mSclSt|mSclSt|10|
OP_mShflBMDst|mShflBMDst|2|
OP_mShflXDst|mShflXDst|25|
OP_mWa|mWa|24|
OP_mWb|mWb|24|
OP_mWm|mWm|24|
OP_mWs|mWs|24|
OP_mXa|mXa|25|
OP_mXb|mXb|25|
OP_mXm|mXm|25|
OP_mXn|mXn|25|
OP_mXs|mXs|25|
OP_mXv|mXv|25|
OP_mXw|mXw|25|
"""

_IMMEDIATE_ENCODING_RECORDS = """\
c10s_step64|10|4|64|1
c11s|11|11|1|1
c12n_step4|12|9|4|3
c14n_step16|14|9|16|3
c15n_step32|15|9|32|3
c16n_step64|16|9|64|3
c19s_step64|19|13|64|1
c1u|1|1|1|0
c2u|2|2|1|0
c32s|32|32|1|1
c3u|3|3|1|0
c4s|4|4|1|1
c5s_step2|5|4|2|1
c5u|5|5|1|0
c6s_step4|6|4|4|1
c6u|6|6|1|0
c7s|7|7|1|1
c8s|8|8|1|1
c8s_step16|8|4|16|1
c9s_step32|9|4|32|1
cpmaddr|20|20|1|4
"""

_MACHINE_FORM_RECORDS = """\
ABS|II_ABS|0||d0:eR|s0:eR|srCarry|||"abs\\t $d0, $s0"
ACQ_COND_mLockId_imm|II_ACQ_COND_mLockId_imm|8|||id:c6u,s1:eR,s2:mR26_lock||||"acq.cond\\t#$id, $s1, r26"
ACQ_COND_mLockId_reg|II_ACQ_COND_mLockId_reg|8|||id:eR,s1:eR,s2:mR26_lock||||"acq.cond\\t$id, $s1, r26"
ACQ_mLockId_imm|II_ACQ_mLockId_imm|8|||id:c6u,s1:eR||||"acq\\t#$id, $s1"
ACQ_mLockId_reg|II_ACQ_mLockId_reg|8|||id:eR,s1:eR||||"acq\\t$id, $s1"
ADC|II_ADC|0||d0:eR|s0:eR,s1:eR|srCarry|srCarry||"adc\\t $d0, $s0, $s1"
ADD_NC_mv_add_ri|II_ADD_NC_mv_add_ri|0||dst:OP_mMvSclDst|s0:eR,imm:c8s||||"add.nc\\t$dst, $s0, #$imm"
ADD_NC_mv_add_rr|II_ADD_NC_mv_add_rr|0||dst:OP_mMvSclDst|s0:eR,s1:eR||||"add.nc\\t$dst, $s0, $s1"
ADD_add_r_ri|II_ADD_add_r_ri|0||d0:eR|s0:eR,imm:c7s|srCarry|||"add\\t$d0, $s0, #$imm"
ADD_alu_r_rr|II_ADD_alu_r_rr|0||d0:eR|s0:eR,s1:eR|srCarry|||"add\\t $d0, $s0, $s1"
AND|II_AND|0||d0:eR|s0:eR,s1:eR||||"and\\t $d0, $s0, $s1"
ASHL|II_ASHL|0||d0:eR|s0:eR,s1:eR||||"ashl\\t $d0, $s0, $s1"
CLB|II_CLB|0||d0:eR|s0:eR||||"clb\\t $d0, $s0"
CLZ|II_CLZ|0||d0:eR|s0:eR||||"clz\\t $d0, $s0"
DIVS|II_DIVS|0||d0:eR,sd_out:mR31_divs|sd:mR31_divs,s0:eR,s1:eR|||sd_out=sd|"divs\\t$d0, r31, $s0, $s1"
DONE|II_DONE|8|||||||"done\\t"
EQ|II_EQ|0||d0:eR|s0:eR,s1:eR||||"eq\\t $d0, $s0, $s1"
EQZ|II_EQZ|0||d0:eR|s0:eR||||"eqz\\t $d0, $s0"
EVENT_ERROR|II_EVENT_ERROR|8|||||||"event.error\\t"
EVENT_WARNING|II_EVENT_WARNING|8|||||||"event.warning\\t"
EVENT_event0|II_EVENT_event0|8|||||||"event\\t#0"
EVENT_event1|II_EVENT_event1|8|||||||"event\\t#1"
EXTEND_s16|II_EXTEND_s16|0||d0:eR|s0:eR||||"extend.s16\\t $d0, $s0"
EXTEND_s8|II_EXTEND_s8|0||d0:eR|s0:eR||||"extend.s8\\t $d0, $s0"
EXTEND_u16|II_EXTEND_u16|0||d0:eR|s0:eR||||"extend.u16\\t $d0, $s0"
EXTEND_u8|II_EXTEND_u8|0||d0:eR|s0:eR||||"extend.u8\\t $d0, $s0"
GE|II_GE|0||d0:eR|s0:eR,s1:eR||||"ge\\t $d0, $s0, $s1"
GEU|II_GEU|0||d0:eR|s0:eR,s1:eR||||"geu\\t $d0, $s0, $s1"
INVSQRT_mOptConvDel_mRx|II_INVSQRT_mOptConvDel_mRx|0||d0:eR|m:mS3,s0:eR|srFPCnvFl2Fx,srFPNlf|crFPCnvFl2FxMask,crFPNlfMask||"invsqrt\\t ${d0}.flt2fx, s3, $s0"
INVSQRT_mOptConvDel_mRx_mOptConv|II_INVSQRT_mOptConvDel_mRx_mOptConv|0||d0:eR|m:mS3,s0:eR,m_1:mS2|srFPCnvFl2Fx,srFPCnvFx2Fl,srFPNlf|crFPCnvFl2FxMask,crFPCnvFx2FlMask,crFPNlfMask||"invsqrt\\t ${d0}.flt2fx, s3, ${s0}.fx2flt, s2"
INVSQRT_mRx|II_INVSQRT_mRx|0||d0:eR|s0:eR|srFPNlf|crFPNlfMask||"invsqrt\\t $d0, $s0"
INVSQRT_mRx_mOptConv|II_INVSQRT_mRx_mOptConv|0||d0:eR|s0:eR,m:mS2|srFPCnvFx2Fl,srFPNlf|crFPCnvFx2FlMask,crFPNlfMask||"invsqrt\\t $d0, ${s0}.fx2flt, s2"
INV_mOptConvDel_mRx|II_INV_mOptConvDel_mRx|0||d0:eR|m:mS3,s0:eR|srFPCnvFl2Fx,srFPNlf|crFPCnvFl2FxMask,crFPNlfMask||"inv\\t ${d0}.flt2fx, s3, $s0"
INV_mOptConvDel_mRx_mOptConv|II_INV_mOptConvDel_mRx_mOptConv|0||d0:eR|m:mS3,s0:eR,m_1:mS2|srFPCnvFl2Fx,srFPCnvFx2Fl,srFPNlf|crFPCnvFl2FxMask,crFPCnvFx2FlMask,crFPNlfMask||"inv\\t ${d0}.flt2fx, s3, ${s0}.fx2flt, s2"
INV_mRx|II_INV_mRx|0||d0:eR|s0:eR|srFPNlf|crFPNlfMask||"inv\\t $d0, $s0"
INV_mRx_mOptConv|II_INV_mRx_mOptConv|0||d0:eR|s0:eR,m:mS2|srFPCnvFx2Fl,srFPNlf|crFPCnvFx2FlMask,crFPNlfMask||"inv\\t $d0, ${s0}.fx2flt, s2"
JL_alumv_or|II_JL_alumv_or|1|call_indirect||a:eP|lr|||"jl\\t$a"
JL_lng|II_JL_lng|1|call_direct||i:cpmaddr|lr|||"jl\\t#$i"
JNZ|II_JNZ|1|branch_conditional_nonzero||s0:eR,i:cpmaddr||||"jnz\\t $s0, #$i"
JNZD|II_JNZD|1|branch_conditional_decrement|d0:eR|s0:eR,a:eP|srCarry|||"jnzd\\t$d0, $s0, $a"
JZ|II_JZ|1|branch_conditional_zero||s0:eR,i:cpmaddr||||"jz\\t $s0, #$i"
J_alumv_or|II_J_alumv_or|1|branch_indirect||a:eP||||"j\\t$a"
J_lng|II_J_lng|1|branch_direct||i:cpmaddr||||"j\\t#$i"
LDA_2D_dms_lda|II_LDA_2D_dms_lda|4006||dst:OP_mLdaScl,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"lda.2d\\t $dst, [$ptr], $mod"
LDA_2D_dmv_lda_q|II_LDA_2D_dmv_lda_q|4006||dst:OP_mQQsa,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"lda.2d\\t $dst, [$ptr], $mod"
LDA_2D_s16|II_LDA_2D_s16|4006||dst:eR,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"lda.2d.s16\\t $dst, [$ptr], $mod"
LDA_2D_s8|II_LDA_2D_s8|4006||dst:eR,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"lda.2d.s8\\t $dst, [$ptr], $mod"
LDA_2D_u16|II_LDA_2D_u16|4006||dst:eR,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"lda.2d.u16\\t $dst, [$ptr], $mod"
LDA_2D_u8|II_LDA_2D_u8|4006||dst:eR,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"lda.2d.u8\\t $dst, [$ptr], $mod"
LDA_3D_dms_lda|II_LDA_3D_dms_lda|4006||dst:OP_mLdaScl,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"lda.3d\\t $dst, [$ptr], $mod"
LDA_3D_dmv_lda_q|II_LDA_3D_dmv_lda_q|4006||dst:OP_mQQsa,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"lda.3d\\t $dst, [$ptr], $mod"
LDA_3D_s16|II_LDA_3D_s16|4006||dst:eR,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"lda.3d.s16\\t $dst, [$ptr], $mod"
LDA_3D_s8|II_LDA_3D_s8|4006||dst:eR,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"lda.3d.s8\\t $dst, [$ptr], $mod"
LDA_3D_u16|II_LDA_3D_u16|4006||dst:eR,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"lda.3d.u16\\t $dst, [$ptr], $mod"
LDA_3D_u8|II_LDA_3D_u8|4006||dst:eR,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"lda.3d.u8\\t $dst, [$ptr], $mod"
LDA_TM_2D|II_LDA_TM_2D|4006||dst:eR,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"lda.tm.2d\\t $dst, [$ptr], $mod"
LDA_TM_3D|II_LDA_TM_3D|4006||dst:eR,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"lda.tm.3d\\t $dst, [$ptr], $mod"
LDA_TM_idx|II_LDA_TM_idx|4000||dst:eR|ptr:eP,dj:eDJ||||"lda.tm\\t $dst, [$ptr, $dj]"
LDA_TM_idx_imm|II_LDA_TM_idx_imm|4000||dst:eR|ptr:eP,imm:c6s_step4||||"lda.tm\\t $dst, [$ptr, #$imm]"
LDA_TM_pstm_nrm|II_LDA_TM_pstm_nrm|4000||dst:eR,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"lda.tm\\t $dst, [$ptr], $mod"
LDA_TM_pstm_nrm_imm|II_LDA_TM_pstm_nrm_imm|4000||dst:eR,ptr_out:eP|ptr:eP,imm:c6s_step4|||ptr_out=ptr|"lda.tm\\t $dst, [$ptr], #$imm"
LDA_dms_lda_idx|II_LDA_dms_lda_idx|4000||dst:OP_mLdaScl|ptr:eP,dj:eDJ||||"lda\\t $dst, [$ptr, $dj]"
LDA_dms_lda_idx_imm|II_LDA_dms_lda_idx_imm|4000||dst:OP_mLdaScl|ptr:eP,imm:c6s_step4||||"lda\\t $dst, [$ptr, #$imm]"
LDA_dms_lda_pstm_nrm|II_LDA_dms_lda_pstm_nrm|4000||dst:OP_mLdaScl,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"lda\\t $dst, [$ptr], $mod"
LDA_dms_lda_pstm_nrm_imm|II_LDA_dms_lda_pstm_nrm_imm|4000||dst:OP_mLdaScl,ptr_out:eP|ptr:eP,imm:c6s_step4|||ptr_out=ptr|"lda\\t $dst, [$ptr], #$imm"
LDA_dms_lda_spill|II_LDA_dms_lda_spill|4000||dst:OP_mLdaScl|imm:c12n_step4||sp||"lda\\t $dst, [sp, #$imm]"
LDA_dmv_lda_q_idx|II_LDA_dmv_lda_q_idx|4000||dst:OP_mQQsa|ptr:eP,dj:eDJ||||"lda\\t $dst, [$ptr, $dj]"
LDA_dmv_lda_q_idx_imm|II_LDA_dmv_lda_q_idx_imm|4000||dst:OP_mQQsa|ptr:eP,imm:c8s_step16||||"lda\\t $dst, [$ptr, #$imm]"
LDA_dmv_lda_q_pstm_nrm|II_LDA_dmv_lda_q_pstm_nrm|4000||dst:OP_mQQsa,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"lda\\t $dst, [$ptr], $mod"
LDA_dmv_lda_q_pstm_nrm_imm|II_LDA_dmv_lda_q_pstm_nrm_imm|4000||dst:OP_mQQsa,ptr_out:eP|ptr:eP,imm:c8s_step16|||ptr_out=ptr|"lda\\t $dst, [$ptr], #$imm"
LDA_dmv_lda_q_spill|II_LDA_dmv_lda_q_spill|4000||dst:OP_mQQsa|imm:c14n_step16||sp||"lda\\t $dst, [sp, #$imm]"
LDA_s16_idx|II_LDA_s16_idx|4000||dst:eR|ptr:eP,dj:eDJ||||"lda.s16\\t $dst, [$ptr, $dj]"
LDA_s16_idx_imm|II_LDA_s16_idx_imm|4000||dst:eR|ptr:eP,imm:c5s_step2||||"lda.s16\\t $dst, [$ptr, #$imm]"
LDA_s16_pstm_nrm|II_LDA_s16_pstm_nrm|4000||dst:eR,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"lda.s16\\t $dst, [$ptr], $mod"
LDA_s16_pstm_nrm_imm|II_LDA_s16_pstm_nrm_imm|4000||dst:eR,ptr_out:eP|ptr:eP,imm:c5s_step2|||ptr_out=ptr|"lda.s16\\t $dst, [$ptr], #$imm"
LDA_s8_idx|II_LDA_s8_idx|4000||dst:eR|ptr:eP,dj:eDJ||||"lda.s8\\t $dst, [$ptr, $dj]"
LDA_s8_idx_imm|II_LDA_s8_idx_imm|4000||dst:eR|ptr:eP,imm:c4s||||"lda.s8\\t $dst, [$ptr, #$imm]"
LDA_s8_pstm_nrm|II_LDA_s8_pstm_nrm|4000||dst:eR,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"lda.s8\\t $dst, [$ptr], $mod"
LDA_s8_pstm_nrm_imm|II_LDA_s8_pstm_nrm_imm|4000||dst:eR,ptr_out:eP|ptr:eP,imm:c4s|||ptr_out=ptr|"lda.s8\\t $dst, [$ptr], #$imm"
LDA_u16_idx|II_LDA_u16_idx|4000||dst:eR|ptr:eP,dj:eDJ||||"lda.u16\\t $dst, [$ptr, $dj]"
LDA_u16_idx_imm|II_LDA_u16_idx_imm|4000||dst:eR|ptr:eP,imm:c5s_step2||||"lda.u16\\t $dst, [$ptr, #$imm]"
LDA_u16_pstm_nrm|II_LDA_u16_pstm_nrm|4000||dst:eR,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"lda.u16\\t $dst, [$ptr], $mod"
LDA_u16_pstm_nrm_imm|II_LDA_u16_pstm_nrm_imm|4000||dst:eR,ptr_out:eP|ptr:eP,imm:c5s_step2|||ptr_out=ptr|"lda.u16\\t $dst, [$ptr], #$imm"
LDA_u8_idx|II_LDA_u8_idx|4000||dst:eR|ptr:eP,dj:eDJ||||"lda.u8\\t $dst, [$ptr, $dj]"
LDA_u8_idx_imm|II_LDA_u8_idx_imm|4000||dst:eR|ptr:eP,imm:c4s||||"lda.u8\\t $dst, [$ptr, #$imm]"
LDA_u8_pstm_nrm|II_LDA_u8_pstm_nrm|4000||dst:eR,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"lda.u8\\t $dst, [$ptr], $mod"
LDA_u8_pstm_nrm_imm|II_LDA_u8_pstm_nrm_imm|4000||dst:eR,ptr_out:eP|ptr:eP,imm:c4s|||ptr_out=ptr|"lda.u8\\t $dst, [$ptr], #$imm"
LSHL|II_LSHL|0||d0:eR|s0:eR,s1:eR||||"lshl\\t $d0, $s0, $s1"
LT|II_LT|0||d0:eR|s0:eR,s1:eR||||"lt\\t $d0, $s0, $s1"
LTU|II_LTU|0||d0:eR|s0:eR,s1:eR||||"ltu\\t $d0, $s0, $s1"
MAC|II_MAC|0||d0:eR|a0:eR,s0:eR,s1:eR|||d0=a0|"mac\\t$d0, $a0, $s0, $s1"
MOVA|II_MOVA|a00||dst:OP_mLdaCg|i:c11s||||"mova\\t$dst, #$i"
MOVS|II_MOVS|0||dst:OP_mAguDst|src:OP_mAguSrc||||"movs\\t$dst, $src"
MOVXM|II_MOVXM|a00||dst:OP_mMvSclDstCg|i:c32s||||"movxm\\t$dst, #$i"
MOVX_alu_cg|II_MOVX_alu_cg|a00||dst:OP_mAluCg|i:c11s||||"movx\\t$dst, #$i"
MOVX_mvx_cr_imm|II_MOVX_mvx_cr_imm|a00||dst:OP_mCRm|src:c5u||||"movx\\t$dst, #$src"
MOVX_mvx_cr_r|II_MOVX_mvx_cr_r|0||dst:OP_mCRm|src:eR||||"movx\\t$dst, $src"
MOV_CPH_mMStream_tlast_imm|II_MOV_CPH_mMStream_tlast_imm|8|||addr:eM,nw:c2u,op:c2u,id:eR|srMS0|||"mov.cph\\t ms, $addr, #$nw, #$op, $id"
MOV_CPH_mMStream_tlast_reg|II_MOV_CPH_mMStream_tlast_reg|8|||addr:eM,nw:c2u,op:c2u,id:eR,dst:mR28_tlast|srMS0|||"mov.cph\\t ms, $addr, #$nw, #$op, $id, r28"
MOV_CPH_nb_mMStream_tlast_imm|II_MOV_CPH_nb_mMStream_tlast_imm|8|||addr:eM,nw:c2u,op:c2u,id:eR|srMS0|||"mov.cph.nb\\t ms, $addr, #$nw, #$op, $id"
MOV_CPH_nb_mMStream_tlast_reg|II_MOV_CPH_nb_mMStream_tlast_reg|8|||addr:eM,nw:c2u,op:c2u,id:eR,dst:mR28_tlast|srMS0|||"mov.cph.nb\\t ms, $addr, #$nw, #$op, $id, r28"
MOV_CPH_nb_tlast|II_MOV_CPH_nb_tlast|8|||addr:eM,nw:c2u,op:c2u,id:eR|srMS0|||"mov.cph.nb.tlast\\t ms, $addr, #$nw, #$op, $id"
MOV_CPH_tlast|II_MOV_CPH_tlast|8|||addr:eM,nw:c2u,op:c2u,id:eR|srMS0|||"mov.cph.tlast\\t ms, $addr, #$nw, #$op, $id"
MOV_PH_mMStream_tlast_imm|II_MOV_PH_mMStream_tlast_imm|8|||id:eR,pcktType:c3u|srMS0|||"mov.ph\\t ms, $id, #$pcktType"
MOV_PH_mMStream_tlast_reg|II_MOV_PH_mMStream_tlast_reg|8|||id:eR,pcktType:c3u,dst:mR28_tlast|srMS0|||"mov.ph\\t ms, $id, #$pcktType, r28"
MOV_PH_nb_mMStream_tlast_imm|II_MOV_PH_nb_mMStream_tlast_imm|8|||id:eR,pcktType:c3u|srMS0|||"mov.ph.nb\\t ms, $id, #$pcktType"
MOV_PH_nb_mMStream_tlast_reg|II_MOV_PH_nb_mMStream_tlast_reg|8|||id:eR,pcktType:c3u,dst:mR28_tlast|srMS0|||"mov.ph.nb\\t ms, $id, #$pcktType, r28"
MOV_PH_nb_tlast|II_MOV_PH_nb_tlast|8|||id:eR,pcktType:c3u|srMS0|||"mov.ph.nb.tlast\\t ms, $id, #$pcktType"
MOV_PH_tlast|II_MOV_PH_tlast|8|||id:eR,pcktType:c3u|srMS0|||"mov.ph.tlast\\t ms, $id, #$pcktType"
MOV_alu_mv_alu_flt2fx|II_MOV_alu_mv_alu_flt2fx|0||d0:eR|s0:eR,m:mS3|srFPCnvFl2Fx|crFPCnvFl2FxMask||"mov\\t$d0, ${s0}.flt2fx, s3"
MOV_alu_mv_alu_fx2flt|II_MOV_alu_mv_alu_fx2flt|0||d0:eR|s0:eR,m:mS2|srFPCnvFx2Fl|crFPCnvFx2FlMask||"mov\\t$d0, ${s0}.fx2flt, s2"
MOV_alu_mv_mv_mv_cg|II_MOV_alu_mv_mv_mv_cg|a00||dst:OP_mMvSclDstCg|i:c11s||||"mov\\t$dst, #$i"
MOV_alu_mv_mv_mv_cntr2l|II_MOV_alu_mv_mv_mv_cntr2l|8||dst:eL|||||"mov\\t$dst, cntr"
MOV_alu_mv_mv_mv_e_mv_eh_to_eh|II_MOV_alu_mv_mv_mv_e_mv_eh_to_eh|0||dst:OP_mEhm|src:OP_mEhm||||"mov\\t$dst, $src"
MOV_alu_mv_mv_mv_e_mv_eh_to_el|II_MOV_alu_mv_mv_mv_e_mv_eh_to_el|0||dst:OP_mElm|src:OP_mEhm||||"mov\\t$dst, $src"
MOV_alu_mv_mv_mv_e_mv_eh_to_r|II_MOV_alu_mv_mv_mv_e_mv_eh_to_r|0||dst:eR|src:OP_mEhm||||"mov\\t$dst, $src"
MOV_alu_mv_mv_mv_e_mv_el_to_eh|II_MOV_alu_mv_mv_mv_e_mv_el_to_eh|0||dst:OP_mEhm|src:OP_mElm||||"mov\\t$dst, $src"
MOV_alu_mv_mv_mv_e_mv_el_to_el|II_MOV_alu_mv_mv_mv_e_mv_el_to_el|0||dst:OP_mElm|src:OP_mElm||||"mov\\t$dst, $src"
MOV_alu_mv_mv_mv_e_mv_el_to_r|II_MOV_alu_mv_mv_mv_e_mv_el_to_r|0||dst:eR|src:OP_mElm||||"mov\\t$dst, $src"
MOV_alu_mv_mv_mv_e_mv_r_to_eh|II_MOV_alu_mv_mv_mv_e_mv_r_to_eh|0||dst:OP_mEhm|src:eR||||"mov\\t$dst, $src"
MOV_alu_mv_mv_mv_e_mv_r_to_el|II_MOV_alu_mv_mv_mv_e_mv_r_to_el|0||dst:OP_mElm|src:eR||||"mov\\t$dst, $src"
MOV_alu_mv_mv_mv_scl|II_MOV_alu_mv_mv_mv_scl|0||dst:OP_mMvSclDst|src:OP_mMvSclSrc||||"mov\\t$dst, $src"
MOV_d1|II_MOV_d1|0||dst:eR|src:OP_mMvSclSrc||||"mov.d1\\t $dst, $src"
MOV_d2|II_MOV_d2|0||dst:eR|src:OP_mMvSclSrc||||"mov.d2\\t $dst, $src"
MOV_d3|II_MOV_d3|0||dst:eR|src:OP_mMvSclSrc||||"mov.d3\\t $dst, $src"
MOV_d4|II_MOV_d4|0||dst:eR|src:OP_mMvSclSrc||||"mov.d4\\t $dst, $src"
MOV_d5|II_MOV_d5|0||dst:eR|src:OP_mMvSclSrc||||"mov.d5\\t $dst, $src"
MOV_d6|II_MOV_d6|0||dst:eR|src:OP_mMvSclSrc||||"mov.d6\\t $dst, $src"
MOV_lda|II_MOV_lda|8||dst:eR||srSS0|||"mov\\t $dst, ss"
MOV_nb_lda|II_MOV_nb_lda|8||dst:eR||srSS0|||"mov.nb\\t $dst, ss"
MOV_nb_st_mMStream_tlast_imm|II_MOV_nb_st_mMStream_tlast_imm|8|||src:OP_mSclMS|srMS0|||"mov.nb\\t ms, $src"
MOV_nb_st_mMStream_tlast_reg|II_MOV_nb_st_mMStream_tlast_reg|8|||src:OP_mSclMS,dst:mR28_tlast|srMS0|||"mov.nb\\t ms, $src, r28"
MOV_nb_tlast|II_MOV_nb_tlast|8|||src:OP_mSclMS|srMS0|||"mov.nb.tlast\\t ms, $src"
MOV_st_mMStream_tlast_imm|II_MOV_st_mMStream_tlast_imm|8|||src:OP_mSclMS|srMS0|||"mov\\t ms, $src"
MOV_st_mMStream_tlast_reg|II_MOV_st_mMStream_tlast_reg|8|||src:OP_mSclMS,dst:mR28_tlast|srMS0|||"mov\\t ms, $src, r28"
MOV_tlast|II_MOV_tlast|8|||src:OP_mSclMS|srMS0|||"mov.tlast\\t ms, $src"
MSC|II_MSC|0||d0:eR|a0:eR,s0:eR,s1:eR|||d0=a0|"msc\\t$d0, $a0, $s0, $s1"
MUL|II_MUL|0||d0:eR|s0:eR,s1:eR||||"mul\\t$d0, $s0, $s1"
NE|II_NE|0||d0:eR|s0:eR,s1:eR||||"ne\\t $d0, $s0, $s1"
NEZ|II_NEZ|0||d0:eR|s0:eR||||"nez\\t $d0, $s0"
NOP|NoItinerary|0|||||||"nop\\t"
NOPA|II_NOPA|0|||||||"nopa\\t"
NOPB|II_NOPB|0|||||||"nopb\\t"
NOPM|II_NOPM|0|||||||"nopm\\t"
NOPS|II_NOPS|0|||||||"nops\\t"
NOPV|II_NOPV|0|||||||"nopv\\t"
NOPX|II_NOPX|0|||||||"nopx\\t"
NOPXM|II_NOPXM|0|||||||"nopxm\\t"
OR|II_OR|0||d0:eR|s0:eR,s1:eR||||"or\\t $d0, $s0, $s1"
PADDA_2D|II_PADDA_2D|6||ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"padda.2d\\t [$ptr], $mod"
PADDA_3D|II_PADDA_3D|6||ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"padda.3d\\t [$ptr], $mod"
PADDA_pstm_nrm|II_PADDA_pstm_nrm|0||ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"padda\\t [$ptr], $mod"
PADDA_pstm_nrm_imm|II_PADDA_pstm_nrm_imm|0||ptr_out:eP|ptr:eP,imm:c10s_step64|||ptr_out=ptr|"padda\\t [$ptr], #$imm"
PADDB_2D|II_PADDB_2D|6||ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"paddb.2d\\t [$ptr], $mod"
PADDB_3D|II_PADDB_3D|6||ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"paddb.3d\\t [$ptr], $mod"
PADDB_pstm_nrm|II_PADDB_pstm_nrm|0||ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"paddb\\t [$ptr], $mod"
PADDB_pstm_nrm_imm|II_PADDB_pstm_nrm_imm|0||ptr_out:eP|ptr:eP,imm:c10s_step64|||ptr_out=ptr|"paddb\\t [$ptr], #$imm"
PADDS_2D|II_PADDS_2D|6||ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"padds.2d\\t [$ptr], $mod"
PADDS_3D|II_PADDS_3D|6||ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"padds.3d\\t [$ptr], $mod"
PADDS_pstm_nrm|II_PADDS_pstm_nrm|0||ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"padds\\t [$ptr], $mod"
PADDS_pstm_nrm_imm|II_PADDS_pstm_nrm_imm|0||ptr_out:eP|ptr:eP,imm:c10s_step64|||ptr_out=ptr|"padds\\t [$ptr], #$imm"
PADDXM_pstm_sp|II_PADDXM_pstm_sp|0|||mod:eM|sp|sp||"paddxm\\t [sp], $mod"
PADDXM_pstm_sp_imm|II_PADDXM_pstm_sp_imm|0|||imm:c19s_step64|sp|sp||"paddxm\\t [sp], #$imm"
POPCOUNT|II_POPCOUNT|0||d0:eR|s0:eR||||"popcount\\t $d0, $s0"
REL_COND_mLockId_imm|II_REL_COND_mLockId_imm|8|||id:c6u,s1:eR,s2:mR26_lock||||"rel.cond\\t#$id, $s1, r26"
REL_COND_mLockId_reg|II_REL_COND_mLockId_reg|8|||id:eR,s1:eR,s2:mR26_lock||||"rel.cond\\t$id, $s1, r26"
REL_mLockId_imm|II_REL_mLockId_imm|8|||id:c6u,s1:eR||||"rel\\t#$id, $s1"
REL_mLockId_reg|II_REL_mLockId_reg|8|||id:eR,s1:eR||||"rel\\t$id, $s1"
RET|II_RET|1|return||||lr||"ret\\tlr"
SBC|II_SBC|0||d0:eR|s0:eR,s1:eR|srCarry|srCarry||"sbc\\t $d0, $s0, $s1"
SEL_EQZ|II_SEL_EQZ|0||d0:eR|s0:eR,s1:eR,s2:mR27_select||||"sel.eqz\\t $d0, $s0, $s1, r27"
SEL_NEZ|II_SEL_NEZ|0||d0:eR|s0:eR,s1:eR,s2:mR27_select||||"sel.nez\\t $d0, $s0, $s1, r27"
SQRT_mOptConvDel_mRx|II_SQRT_mOptConvDel_mRx|0||d0:eR|m:mS3,s0:eR|srFPCnvFl2Fx,srFPNlf|crFPCnvFl2FxMask,crFPNlfMask||"sqrt\\t ${d0}.flt2fx, s3, $s0"
SQRT_mOptConvDel_mRx_mOptConv|II_SQRT_mOptConvDel_mRx_mOptConv|0||d0:eR|m:mS3,s0:eR,m_1:mS2|srFPCnvFl2Fx,srFPCnvFx2Fl,srFPNlf|crFPCnvFl2FxMask,crFPCnvFx2FlMask,crFPNlfMask||"sqrt\\t ${d0}.flt2fx, s3, ${s0}.fx2flt, s2"
SQRT_mRx|II_SQRT_mRx|0||d0:eR|s0:eR|srFPNlf|crFPNlfMask||"sqrt\\t $d0, $s0"
SQRT_mRx_mOptConv|II_SQRT_mRx_mOptConv|0||d0:eR|s0:eR,m:mS2|srFPCnvFx2Fl,srFPNlf|crFPCnvFx2FlMask,crFPNlfMask||"sqrt\\t $d0, ${s0}.fx2flt, s2"
ST_2D_dms_sts|II_ST_2D_dms_sts|8006||ptr_out:eP,dc:eDC|src:OP_mSclSt,ptr:eP,mod:eD|||ptr_out=ptr|"st.2d\\t $src, [$ptr], $mod"
ST_2D_dmv_sts_q|II_ST_2D_dmv_sts_q|8006||ptr_out:eP,dc:eDC|src:OP_mQQss,ptr:eP,mod:eD|||ptr_out=ptr|"st.2d\\t $src, [$ptr], $mod"
ST_2D_s16|II_ST_2D_s16|c006||ptr_out:eP,dc:eDC|src:eR,ptr:eP,mod:eD|pe2_ads|pe2_ads|ptr_out=ptr|"st.2d.s16\\t $src, [$ptr], $mod"
ST_2D_s8|II_ST_2D_s8|c006||ptr_out:eP,dc:eDC|src:eR,ptr:eP,mod:eD|pe2_ads|pe2_ads|ptr_out=ptr|"st.2d.s8\\t $src, [$ptr], $mod"
ST_3D_dms_sts|II_ST_3D_dms_sts|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mSclSt,ptr:eP,mod:eDS|||ptr_out=ptr|"st.3d\\t $src, [$ptr], $mod"
ST_3D_dmv_sts_q|II_ST_3D_dmv_sts_q|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mQQss,ptr:eP,mod:eDS|||ptr_out=ptr|"st.3d\\t $src, [$ptr], $mod"
ST_3D_s16|II_ST_3D_s16|c006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:eR,ptr:eP,mod:eDS|pe2_ads|pe2_ads|ptr_out=ptr|"st.3d.s16\\t $src, [$ptr], $mod"
ST_3D_s8|II_ST_3D_s8|c006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:eR,ptr:eP,mod:eDS|pe2_ads|pe2_ads|ptr_out=ptr|"st.3d.s8\\t $src, [$ptr], $mod"
ST_TM_2D|II_ST_TM_2D|8006||ptr_out:eP,dc:eDC|src:eR,ptr:eP,mod:eD|||ptr_out=ptr|"st.tm.2d\\t $src, [$ptr], $mod"
ST_TM_3D|II_ST_TM_3D|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:eR,ptr:eP,mod:eDS|||ptr_out=ptr|"st.tm.3d\\t $src, [$ptr], $mod"
ST_TM_idx|II_ST_TM_idx|8000|||src:eR,ptr:eP,dj:eDJ||||"st.tm\\t $src, [$ptr, $dj]"
ST_TM_idx_imm|II_ST_TM_idx_imm|8000|||src:eR,ptr:eP,imm:c6s_step4||||"st.tm\\t $src, [$ptr, #$imm]"
ST_TM_pstm_nrm|II_ST_TM_pstm_nrm|8000||ptr_out:eP|src:eR,ptr:eP,mod:eM|||ptr_out=ptr|"st.tm\\t $src, [$ptr], $mod"
ST_TM_pstm_nrm_imm|II_ST_TM_pstm_nrm_imm|8000||ptr_out:eP|src:eR,ptr:eP,imm:c6s_step4|||ptr_out=ptr|"st.tm\\t $src, [$ptr], #$imm"
ST_dms_sts_idx|II_ST_dms_sts_idx|8000|||src:OP_mSclSt,ptr:eP,dj:eDJ||||"st\\t $src, [$ptr, $dj]"
ST_dms_sts_idx_imm|II_ST_dms_sts_idx_imm|8000|||src:OP_mSclSt,ptr:eP,imm:c6s_step4||||"st\\t $src, [$ptr, #$imm]"
ST_dms_sts_pstm_nrm|II_ST_dms_sts_pstm_nrm|8000||ptr_out:eP|src:OP_mSclSt,ptr:eP,mod:eM|||ptr_out=ptr|"st\\t $src, [$ptr], $mod"
ST_dms_sts_pstm_nrm_imm|II_ST_dms_sts_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mSclSt,ptr:eP,imm:c6s_step4|||ptr_out=ptr|"st\\t $src, [$ptr], #$imm"
ST_dms_sts_spill|II_ST_dms_sts_spill|8000|||src:OP_mSclSt,imm:c12n_step4||sp||"st\\t $src, [sp, #$imm]"
ST_dmv_sts_q_idx|II_ST_dmv_sts_q_idx|8000|||src:OP_mQQss,ptr:eP,dj:eDJ||||"st\\t $src, [$ptr, $dj]"
ST_dmv_sts_q_idx_imm|II_ST_dmv_sts_q_idx_imm|8000|||src:OP_mQQss,ptr:eP,imm:c8s_step16||||"st\\t $src, [$ptr, #$imm]"
ST_dmv_sts_q_pstm_nrm|II_ST_dmv_sts_q_pstm_nrm|8000||ptr_out:eP|src:OP_mQQss,ptr:eP,mod:eM|||ptr_out=ptr|"st\\t $src, [$ptr], $mod"
ST_dmv_sts_q_pstm_nrm_imm|II_ST_dmv_sts_q_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mQQss,ptr:eP,imm:c8s_step16|||ptr_out=ptr|"st\\t $src, [$ptr], #$imm"
ST_dmv_sts_q_spill|II_ST_dmv_sts_q_spill|8000|||src:OP_mQQss,imm:c14n_step16||sp||"st\\t $src, [sp, #$imm]"
ST_s16_idx|II_ST_s16_idx|c000|||src:eR,ptr:eP,dj:eDJ|pe2_ads|pe2_ads||"st.s16\\t $src, [$ptr, $dj]"
ST_s16_idx_imm|II_ST_s16_idx_imm|c000|||src:eR,ptr:eP,imm:c5s_step2|pe2_ads|pe2_ads||"st.s16\\t $src, [$ptr, #$imm]"
ST_s16_pstm_nrm|II_ST_s16_pstm_nrm|c000||ptr_out:eP|src:eR,ptr:eP,mod:eM|pe2_ads|pe2_ads|ptr_out=ptr|"st.s16\\t $src, [$ptr], $mod"
ST_s16_pstm_nrm_imm|II_ST_s16_pstm_nrm_imm|c000||ptr_out:eP|src:eR,ptr:eP,imm:c5s_step2|pe2_ads|pe2_ads|ptr_out=ptr|"st.s16\\t $src, [$ptr], #$imm"
ST_s8_idx|II_ST_s8_idx|c000|||src:eR,ptr:eP,dj:eDJ|pe2_ads|pe2_ads||"st.s8\\t $src, [$ptr, $dj]"
ST_s8_idx_imm|II_ST_s8_idx_imm|c000|||src:eR,ptr:eP,imm:c4s|pe2_ads|pe2_ads||"st.s8\\t $src, [$ptr, #$imm]"
ST_s8_pstm_nrm|II_ST_s8_pstm_nrm|c000||ptr_out:eP|src:eR,ptr:eP,mod:eM|pe2_ads|pe2_ads|ptr_out=ptr|"st.s8\\t $src, [$ptr], $mod"
ST_s8_pstm_nrm_imm|II_ST_s8_pstm_nrm_imm|c000||ptr_out:eP|src:eR,ptr:eP,imm:c4s|pe2_ads|pe2_ads|ptr_out=ptr|"st.s8\\t $src, [$ptr], #$imm"
SUB|II_SUB|0||d0:eR|s0:eR,s1:eR|srCarry|||"sub\\t $d0, $s0, $s1"
VABS_GTZ_16_vaddSign0|II_VABS_GTZ_16_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s2:OP_mXn||vaddSign0||"vabs_gtz.16\\t $d, r16, $s2, vaddsign0"
VABS_GTZ_16_vaddSign1|II_VABS_GTZ_16_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s2:OP_mXn||vaddSign1||"vabs_gtz.16\\t $d, r16, $s2, vaddsign1"
VABS_GTZ_32_vaddSign0|II_VABS_GTZ_32_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s2:OP_mXn||vaddSign0||"vabs_gtz.32\\t $d, r16, $s2, vaddsign0"
VABS_GTZ_32_vaddSign1|II_VABS_GTZ_32_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s2:OP_mXn||vaddSign1||"vabs_gtz.32\\t $d, r16, $s2, vaddsign1"
VABS_GTZ_8_vaddSign0|II_VABS_GTZ_8_vaddSign0|0||d:OP_mXm,cmp:mL8m|s2:OP_mXn||vaddSign0||"vabs_gtz.8\\t $d, r17:r16, $s2, vaddsign0"
VABS_GTZ_8_vaddSign1|II_VABS_GTZ_8_vaddSign1|0||d:OP_mXm,cmp:mL8m|s2:OP_mXn||vaddSign1||"vabs_gtz.8\\t $d, r17:r16, $s2, vaddsign1"
VADDMAC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_X_X|II_VADDMAC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_X_X|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask|dst=acc1|"vaddmac.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMAC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_Y_Y|II_VADDMAC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_Y_Y|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask|dst=acc1|"vaddmac.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_X_X|II_VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_X_X|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmac.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_Y_Y|II_VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_Y_Y|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmac.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_X_X|II_VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_X_X|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1|"vaddmac.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_Y_Y|II_VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_Y_Y|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1|"vaddmac.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EX|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EX|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask|dst=acc1|"vaddmac.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EY|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EY|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask|dst=acc1|"vaddmac.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_QEY|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_QEY|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask|dst=acc1|"vaddmac.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EY_QEX|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EY_QEX|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask|dst=acc1|"vaddmac.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EX|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EX|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmac.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EY|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EY|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmac.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_QEY|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_QEY|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmac.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EY_QEX|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EY_QEX|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmac.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EX|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EX|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1|"vaddmac.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EY|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EY|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1|"vaddmac.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_QEY|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_QEY|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask,crSCDEn|dst=acc1|"vaddmac.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EY_QEX|II_VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EY_QEX|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask,crSCDEn|dst=acc1|"vaddmac.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_QX|II_VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_QX|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of||dst=acc1|"vaddmac\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_QY|II_VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_QY|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of||dst=acc1|"vaddmac\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X|II_VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:OP_mXw,acc:eR|||dst=acc1|"vaddmac\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_Y|II_VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_Y|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:eY,acc:eR|||dst=acc1|"vaddmac\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_QX|II_VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_QX|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of||dst=acc1|"vaddmac\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_QY|II_VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_QY|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:eQYs,acc:eR|srSparse_of||dst=acc1|"vaddmac\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_X|II_VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_X|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:OP_mXw,acc:eR|||dst=acc1|"vaddmac\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_Y|II_VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_Y|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:eY,acc:eR|||dst=acc1|"vaddmac\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QX|II_VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QX|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of|crSCDEn|dst=acc1,r_out=r|"vaddmac\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QY|II_VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QY|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of|crSCDEn|dst=acc1,r_out=r|"vaddmac\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_X|II_VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_X|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:OP_mXw,acc:eR||crSCDEn|dst=acc1,r_out=r|"vaddmac\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_Y|II_VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_Y|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:eY,acc:eR||crSCDEn|dst=acc1,r_out=r|"vaddmac\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QX|II_VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QX|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of|crSCDEn|dst=acc1,r_out=r|"vaddmac\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QY|II_VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QY|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:eQYs,acc:eR|srSparse_of|crSCDEn|dst=acc1,r_out=r|"vaddmac\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_X|II_VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_X|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:OP_mXw,acc:eR||crSCDEn|dst=acc1,r_out=r|"vaddmac\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_Y|II_VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_Y|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:eY,acc:eR||crSCDEn|dst=acc1,r_out=r|"vaddmac\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_QX|II_VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_QX|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of|crSCDEn|dst=acc1|"vaddmac\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_QY|II_VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_QY|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of|crSCDEn|dst=acc1|"vaddmac\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_X|II_VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_X|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:OP_mXw,acc:eR||crSCDEn|dst=acc1|"vaddmac\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_Y|II_VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_Y|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:eY,acc:eR||crSCDEn|dst=acc1|"vaddmac\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_QX|II_VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_QX|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of|crSCDEn|dst=acc1|"vaddmac\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_QY|II_VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_QY|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:eQYs,acc:eR|srSparse_of|crSCDEn|dst=acc1|"vaddmac\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_X|II_VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_X|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:OP_mXw,acc:eR||crSCDEn|dst=acc1|"vaddmac\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_Y|II_VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_Y|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:eY,acc:eR||crSCDEn|dst=acc1|"vaddmac\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMSC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_X_X|II_VADDMSC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_X_X|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask|dst=acc1|"vaddmsc.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMSC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_Y_Y|II_VADDMSC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_Y_Y|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask|dst=acc1|"vaddmsc.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_X_X|II_VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_X_X|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmsc.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_Y_Y|II_VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_Y_Y|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmsc.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_X_X|II_VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_X_X|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1|"vaddmsc.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_Y_Y|II_VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_Y_Y|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1|"vaddmsc.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EX|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EX|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask|dst=acc1|"vaddmsc.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EY|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EY|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask|dst=acc1|"vaddmsc.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_QEY|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_QEY|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask|dst=acc1|"vaddmsc.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EY_QEX|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EY_QEX|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask|dst=acc1|"vaddmsc.f\\t$dst, $acc1, $acc2, $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EX|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EX|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmsc.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EY|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EY|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmsc.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_QEY|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_QEY|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmsc.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EY_QEX|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EY_QEX|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask,crSCDEn|dst=acc1,r_out=r|"vaddmsc.f\\t$dst, $acc1, scd[r31++], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EX|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EX|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1|"vaddmsc.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EY|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EY|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask,crSCDEn|dst=acc1|"vaddmsc.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_QEY|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_QEY|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask,crSCDEn|dst=acc1|"vaddmsc.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EY_QEX|II_VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EY_QEX|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask,crSCDEn|dst=acc1|"vaddmsc.f\\t$dst, $acc1, scd[r31], $s1, $s2, $acc"
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_QX|II_VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_QX|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of||dst=acc1|"vaddmsc\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_QY|II_VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_QY|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of||dst=acc1|"vaddmsc\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_X|II_VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_X|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:OP_mXw,acc:eR|||dst=acc1|"vaddmsc\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_Y|II_VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_Y|0||dst:eDM|acc1:eDM,acc2:eDM,s1:OP_mXv,s2:eY,acc:eR|||dst=acc1|"vaddmsc\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_QX|II_VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_QX|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of||dst=acc1|"vaddmsc\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_QY|II_VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_QY|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:eQYs,acc:eR|srSparse_of||dst=acc1|"vaddmsc\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_X|II_VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_X|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:OP_mXw,acc:eR|||dst=acc1|"vaddmsc\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_Y|II_VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_Y|0||dst:eDM|acc1:eDM,acc2:eDM,s1:eY,s2:eY,acc:eR|||dst=acc1|"vaddmsc\\t$dst, $acc1, $acc2, $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QX|II_VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QX|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of|crSCDEn|dst=acc1,r_out=r|"vaddmsc\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QY|II_VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QY|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of|crSCDEn|dst=acc1,r_out=r|"vaddmsc\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_X|II_VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_X|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:OP_mXw,acc:eR||crSCDEn|dst=acc1,r_out=r|"vaddmsc\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_Y|II_VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_Y|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:OP_mXv,s2:eY,acc:eR||crSCDEn|dst=acc1,r_out=r|"vaddmsc\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QX|II_VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QX|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of|crSCDEn|dst=acc1,r_out=r|"vaddmsc\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QY|II_VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QY|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:eQYs,acc:eR|srSparse_of|crSCDEn|dst=acc1,r_out=r|"vaddmsc\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_X|II_VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_X|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:OP_mXw,acc:eR||crSCDEn|dst=acc1,r_out=r|"vaddmsc\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_Y|II_VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_Y|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,s1:eY,s2:eY,acc:eR||crSCDEn|dst=acc1,r_out=r|"vaddmsc\\t$dst, $acc1, scd[r31++], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_QX|II_VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_QX|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of|crSCDEn|dst=acc1|"vaddmsc\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_QY|II_VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_QY|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of|crSCDEn|dst=acc1|"vaddmsc\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_X|II_VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_X|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:OP_mXw,acc:eR||crSCDEn|dst=acc1|"vaddmsc\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_Y|II_VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_Y|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:OP_mXv,s2:eY,acc:eR||crSCDEn|dst=acc1|"vaddmsc\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_QX|II_VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_QX|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of|crSCDEn|dst=acc1|"vaddmsc\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_QY|II_VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_QY|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:eQYs,acc:eR|srSparse_of|crSCDEn|dst=acc1|"vaddmsc\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_X|II_VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_X|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:OP_mXw,acc:eR||crSCDEn|dst=acc1|"vaddmsc\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_Y|II_VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_Y|8||dst:eDM|acc1:eDM,c:mR31_scd,s1:eY,s2:eY,acc:eR||crSCDEn|dst=acc1|"vaddmsc\\t$dst, $acc1, scd[r31], $s1, $s2,$acc"
VADDSUB_16|II_VADDSUB_16|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn,sel:eRS16||||"vaddsub.16\\t $d, $s1, $s2, $sel"
VADDSUB_32|II_VADDSUB_32|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn,sel:eRS16||||"vaddsub.32\\t $d, $s1, $s2, $sel"
VADDSUB_8|II_VADDSUB_8|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn,sel:eL||||"vaddsub.8\\t $d, $s1, $s2, $sel"
VADD_16|II_VADD_16|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn||||"vadd.16\\t $d, $s1, $s2"
VADD_32|II_VADD_32|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn||||"vadd.32\\t $d, $s1, $s2"
VADD_8|II_VADD_8|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn||||"vadd.8\\t $d, $s1, $s2"
VADD_f_vmac_cm2_add_reg|II_VADD_f_vmac_cm2_add_reg|0||dst:eDM|acc1:eDM,acc2:eDM,acc:eR|srFPFlags|crFPMask||"vadd.f\\t$dst, $acc1, $acc2, $acc"
VADD_f_vmac_cm2_add_scd|II_VADD_f_vmac_cm2_add_scd|8||dst:eDM|acc1:eDM,c:mR31_scd,acc:eR|srFPFlags|crFPMask,crSCDEn||"vadd.f\\t$dst, $acc1, scd[r31], $acc"
VADD_f_vmac_cm2_add_scd_incr|II_VADD_f_vmac_cm2_add_scd_incr|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,acc:eR|srFPFlags|crFPMask,crSCDEn|r_out=r|"vadd.f\\t$dst, $acc1, scd[r31++], $acc"
VADD_vmac_cm2_add_reg|II_VADD_vmac_cm2_add_reg|0||dst:eDM|acc1:eDM,acc2:eDM,acc:eR||||"vadd\\t$dst, $acc1, $acc2, $acc"
VADD_vmac_cm2_add_scd|II_VADD_vmac_cm2_add_scd|8||dst:eDM|acc1:eDM,c:mR31_scd,acc:eR||crSCDEn||"vadd\\t$dst, $acc1, scd[r31], $acc"
VADD_vmac_cm2_add_scd_incr|II_VADD_vmac_cm2_add_scd_incr|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,acc:eR||crSCDEn|r_out=r|"vadd\\t$dst, $acc1, scd[r31++], $acc"
VBAND|II_VBAND|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn||||"vband\\t $d, $s1, $s2"
VBCSTSHFL_16_vec_broadcast_shuffle_bm|II_VBCSTSHFL_16_vec_broadcast_shuffle_bm|0||dst:OP_mShflBMDst|src:eR,mod:mR29_insert||||"vbcstshfl.16\\t $dst, $src, r29"
VBCSTSHFL_16_vec_broadcast_shuffle_x|II_VBCSTSHFL_16_vec_broadcast_shuffle_x|0||dst:OP_mShflXDst|src:eR,mod:mR29_insert||||"vbcstshfl.16\\t $dst, $src, r29"
VBCSTSHFL_32_vec_broadcast_shuffle_bm|II_VBCSTSHFL_32_vec_broadcast_shuffle_bm|0||dst:OP_mShflBMDst|src:eR,mod:mR29_insert||||"vbcstshfl.32\\t $dst, $src, r29"
VBCSTSHFL_32_vec_broadcast_shuffle_x|II_VBCSTSHFL_32_vec_broadcast_shuffle_x|0||dst:OP_mShflXDst|src:eR,mod:mR29_insert||||"vbcstshfl.32\\t $dst, $src, r29"
VBCSTSHFL_64_vec_broadcast_shuffle_bm|II_VBCSTSHFL_64_vec_broadcast_shuffle_bm|0||dst:OP_mShflBMDst|src:eL,mod:mR29_insert||||"vbcstshfl.64\\t $dst, $src, r29"
VBCSTSHFL_64_vec_broadcast_shuffle_x|II_VBCSTSHFL_64_vec_broadcast_shuffle_x|0||dst:OP_mShflXDst|src:eL,mod:mR29_insert||||"vbcstshfl.64\\t $dst, $src, r29"
VBCSTSHFL_8_vec_broadcast_shuffle_bm|II_VBCSTSHFL_8_vec_broadcast_shuffle_bm|0||dst:OP_mShflBMDst|src:eR,mod:mR29_insert||||"vbcstshfl.8\\t $dst, $src, r29"
VBCSTSHFL_8_vec_broadcast_shuffle_x|II_VBCSTSHFL_8_vec_broadcast_shuffle_x|0||dst:OP_mShflXDst|src:eR,mod:mR29_insert||||"vbcstshfl.8\\t $dst, $src, r29"
VBCST_16|II_VBCST_16|0||dst:OP_mXm|src:eR||||"vbcst.16\\t $dst, $src"
VBCST_32|II_VBCST_32|0||dst:OP_mXm|src:eR||||"vbcst.32\\t $dst, $src"
VBCST_64|II_VBCST_64|0||dst:OP_mXm|src:eL||||"vbcst.64\\t $dst, $src"
VBCST_8|II_VBCST_8|0||dst:OP_mXm|src:eR||||"vbcst.8\\t $dst, $src"
VBNEG_LTZ_s16|II_VBNEG_LTZ_s16|0||d:OP_mXm,cmp:mR16_vcompare|s2:OP_mXn||||"vbneg_ltz.s16\\t $d, r16, $s2"
VBNEG_LTZ_s32|II_VBNEG_LTZ_s32|0||d:OP_mXm,cmp:mR16_vcompare|s2:OP_mXn||||"vbneg_ltz.s32\\t $d, r16, $s2"
VBNEG_LTZ_s8|II_VBNEG_LTZ_s8|0||d:OP_mXm,cmp:mL8m|s2:OP_mXn||||"vbneg_ltz.s8\\t $d, r17:r16, $s2"
VBOR|II_VBOR|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn||||"vbor\\t $d, $s1, $s2"
VCLR|II_VCLR|0||dst:eDM|||||"vclr\\t$dst"
VCONV_bf16_fp32_mv_w_srs_bf|II_VCONV_bf16_fp32_mv_w_srs_bf|0||dst:OP_mWa|src:OP_mBMs|srF2FFlags|crF2FMask,crRnd||"vconv.bf16.fp32\\t $dst, $src"
VCONV_bf16_fp32_mv_x_srs_bf|II_VCONV_bf16_fp32_mv_x_srs_bf|0||dst:OP_mXa|src:OP_mCMs|srF2FFlags|crF2FMask,crRnd||"vconv.bf16.fp32\\t $dst, $src"
VCONV_bfp16ebs16_ebs8|II_VCONV_bfp16ebs16_ebs8|0||dst:OP_mEXa|src:OP_mEXs|srF2BFlags|crF2BMask,crRnd||"vconv.bfp16ebs16.ebs8\\t $dst, $src"
VCONV_bfp16ebs16_fp32|II_VCONV_bfp16ebs16_fp32|0||dst:OP_mEXa|src:eDM|srF2BFlags|crF2BMask,crRnd||"vconv.bfp16ebs16.fp32\\t $dst, $src"
VCONV_bfp16ebs8_fp32|II_VCONV_bfp16ebs8_fp32|0||dst:OP_mEXa|src:eDM|srF2BFlags|crF2BMask,crRnd||"vconv.bfp16ebs8.fp32\\t $dst, $src"
VCONV_fp32_bf16_mv_ups_wbf|II_VCONV_fp32_bf16_mv_ups_wbf|0||dst:OP_mBMm|src:OP_mWm||||"vconv.fp32.bf16\\t$dst, $src"
VCONV_fp32_bf16_mv_ups_xbf|II_VCONV_fp32_bf16_mv_ups_xbf|0||dst:OP_mCMm|src:OP_mXm||||"vconv.fp32.bf16\\t$dst, $src"
VEQZ_16|II_VEQZ_16|0||cmp:eRS16|s2:OP_mXn||||"veqz.16\\t $cmp, $s2"
VEQZ_32|II_VEQZ_32|0||cmp:eRS16|s2:OP_mXn||||"veqz.32\\t $cmp, $s2"
VEQZ_8|II_VEQZ_8|0||cmp:eL|s2:OP_mXn||||"veqz.8\\t $cmp, $s2"
VEXP2|II_VEXP2|0||dst:OP_mWm|src:OP_mBMm||||"vexp2\\t $dst, $src"
VEXTBCSTSHFL_128|II_VEXTBCSTSHFL_128|0||dst:OP_mShflXDst|s1:OP_mXm,idx:eR,mod:mR29_insert||||"vextbcstshfl.128\\t $dst, $s1, $idx, r29"
VEXTBCSTSHFL_16|II_VEXTBCSTSHFL_16|0||dst:OP_mShflXDst|s1:OP_mXm,idx:eR,mod:mR29_insert||||"vextbcstshfl.16\\t $dst, $s1, $idx, r29"
VEXTBCSTSHFL_32|II_VEXTBCSTSHFL_32|0||dst:OP_mShflXDst|s1:OP_mXm,idx:eR,mod:mR29_insert||||"vextbcstshfl.32\\t $dst, $s1, $idx, r29"
VEXTBCSTSHFL_64|II_VEXTBCSTSHFL_64|0||dst:OP_mShflXDst|s1:OP_mXm,idx:eR,mod:mR29_insert||||"vextbcstshfl.64\\t $dst, $s1, $idx, r29"
VEXTBCST_128_vec_extract_broadcast_imm|II_VEXTBCST_128_vec_extract_broadcast_imm|0||dst:OP_mXm|s1:OP_mXm,idx:c6u||||"vextbcst.128\\t $dst, $s1, #$idx"
VEXTBCST_128_vec_extract_broadcast_r|II_VEXTBCST_128_vec_extract_broadcast_r|0||dst:OP_mXm|s1:OP_mXm,idx:eR||||"vextbcst.128\\t $dst, $s1, $idx"
VEXTBCST_16_vec_extract_broadcast_imm|II_VEXTBCST_16_vec_extract_broadcast_imm|0||dst:OP_mXm|s1:OP_mXm,idx:c6u||||"vextbcst.16\\t $dst, $s1, #$idx"
VEXTBCST_16_vec_extract_broadcast_r|II_VEXTBCST_16_vec_extract_broadcast_r|0||dst:OP_mXm|s1:OP_mXm,idx:eR||||"vextbcst.16\\t $dst, $s1, $idx"
VEXTBCST_32_vec_extract_broadcast_imm|II_VEXTBCST_32_vec_extract_broadcast_imm|0||dst:OP_mXm|s1:OP_mXm,idx:c6u||||"vextbcst.32\\t $dst, $s1, #$idx"
VEXTBCST_32_vec_extract_broadcast_r|II_VEXTBCST_32_vec_extract_broadcast_r|0||dst:OP_mXm|s1:OP_mXm,idx:eR||||"vextbcst.32\\t $dst, $s1, $idx"
VEXTBCST_64_vec_extract_broadcast_imm|II_VEXTBCST_64_vec_extract_broadcast_imm|0||dst:OP_mXm|s1:OP_mXm,idx:c6u||||"vextbcst.64\\t $dst, $s1, #$idx"
VEXTBCST_64_vec_extract_broadcast_r|II_VEXTBCST_64_vec_extract_broadcast_r|0||dst:OP_mXm|s1:OP_mXm,idx:eR||||"vextbcst.64\\t $dst, $s1, $idx"
VEXTBCST_8_vec_extract_broadcast_imm|II_VEXTBCST_8_vec_extract_broadcast_imm|0||dst:OP_mXm|s1:OP_mXm,idx:c6u||||"vextbcst.8\\t $dst, $s1, #$idx"
VEXTBCST_8_vec_extract_broadcast_r|II_VEXTBCST_8_vec_extract_broadcast_r|0||dst:OP_mXm|s1:OP_mXm,idx:eR||||"vextbcst.8\\t $dst, $s1, $idx"
VEXTRACT_16_vec_extract_imm_vaddSign0|II_VEXTRACT_16_vec_extract_imm_vaddSign0|0||dst:eR|s1:OP_mXm,idx:c6u||vaddSign0||"vextract.16\\t $dst, $s1, #$idx, vaddsign0"
VEXTRACT_16_vec_extract_imm_vaddSign1|II_VEXTRACT_16_vec_extract_imm_vaddSign1|0||dst:eR|s1:OP_mXm,idx:c6u||vaddSign1||"vextract.16\\t $dst, $s1, #$idx, vaddsign1"
VEXTRACT_16_vec_extract_r_vaddSign0|II_VEXTRACT_16_vec_extract_r_vaddSign0|0||dst:eR|s1:OP_mXm,idx:eR||vaddSign0||"vextract.16\\t $dst, $s1, $idx, vaddsign0"
VEXTRACT_16_vec_extract_r_vaddSign1|II_VEXTRACT_16_vec_extract_r_vaddSign1|0||dst:eR|s1:OP_mXm,idx:eR||vaddSign1||"vextract.16\\t $dst, $s1, $idx, vaddsign1"
VEXTRACT_32_vec_extract_imm_vaddSign0|II_VEXTRACT_32_vec_extract_imm_vaddSign0|0||dst:eR|s1:OP_mXm,idx:c6u||vaddSign0||"vextract.32\\t $dst, $s1, #$idx, vaddsign0"
VEXTRACT_32_vec_extract_imm_vaddSign1|II_VEXTRACT_32_vec_extract_imm_vaddSign1|0||dst:eR|s1:OP_mXm,idx:c6u||vaddSign1||"vextract.32\\t $dst, $s1, #$idx, vaddsign1"
VEXTRACT_32_vec_extract_r_vaddSign0|II_VEXTRACT_32_vec_extract_r_vaddSign0|0||dst:eR|s1:OP_mXm,idx:eR||vaddSign0||"vextract.32\\t $dst, $s1, $idx, vaddsign0"
VEXTRACT_32_vec_extract_r_vaddSign1|II_VEXTRACT_32_vec_extract_r_vaddSign1|0||dst:eR|s1:OP_mXm,idx:eR||vaddSign1||"vextract.32\\t $dst, $s1, $idx, vaddsign1"
VEXTRACT_64_vec_extract_imm_vaddSign0|II_VEXTRACT_64_vec_extract_imm_vaddSign0|0||dst:eL|s1:OP_mXm,idx:c6u||vaddSign0||"vextract.64\\t $dst, $s1, #$idx, vaddsign0"
VEXTRACT_64_vec_extract_imm_vaddSign1|II_VEXTRACT_64_vec_extract_imm_vaddSign1|0||dst:eL|s1:OP_mXm,idx:c6u||vaddSign1||"vextract.64\\t $dst, $s1, #$idx, vaddsign1"
VEXTRACT_64_vec_extract_r_vaddSign0|II_VEXTRACT_64_vec_extract_r_vaddSign0|0||dst:eL|s1:OP_mXm,idx:eR||vaddSign0||"vextract.64\\t $dst, $s1, $idx, vaddsign0"
VEXTRACT_64_vec_extract_r_vaddSign1|II_VEXTRACT_64_vec_extract_r_vaddSign1|0||dst:eL|s1:OP_mXm,idx:eR||vaddSign1||"vextract.64\\t $dst, $s1, $idx, vaddsign1"
VEXTRACT_8_vec_extract_imm_vaddSign0|II_VEXTRACT_8_vec_extract_imm_vaddSign0|0||dst:eR|s1:OP_mXm,idx:c6u||vaddSign0||"vextract.8\\t $dst, $s1, #$idx, vaddsign0"
VEXTRACT_8_vec_extract_imm_vaddSign1|II_VEXTRACT_8_vec_extract_imm_vaddSign1|0||dst:eR|s1:OP_mXm,idx:c6u||vaddSign1||"vextract.8\\t $dst, $s1, #$idx, vaddsign1"
VEXTRACT_8_vec_extract_r_vaddSign0|II_VEXTRACT_8_vec_extract_r_vaddSign0|0||dst:eR|s1:OP_mXm,idx:eR||vaddSign0||"vextract.8\\t $dst, $s1, $idx, vaddsign0"
VEXTRACT_8_vec_extract_r_vaddSign1|II_VEXTRACT_8_vec_extract_r_vaddSign1|0||dst:eR|s1:OP_mXm,idx:eR||vaddSign1||"vextract.8\\t $dst, $s1, $idx, vaddsign1"
VFLOOR_s32_bf16_mv_float_to_int_bm|II_VFLOOR_s32_bf16_mv_float_to_int_bm|0||dst:OP_mXa|src:OP_mBMs,imm:c1u,shft:eS|srF2IFlags|crF2IMask||"vfloor.s32.bf16\\t$dst, $src[$imm], $shft"
VFLOOR_s32_bf16_mv_float_to_int_w|II_VFLOOR_s32_bf16_mv_float_to_int_w|0||dst:OP_mXa|src:OP_mFl2FxSrc_W,shft:eS|srF2IFlags|crF2IMask||"vfloor.s32.bf16\\t$dst, $src, $shft"
VGE_16_vaddSign0|II_VGE_16_vaddSign0|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vge.16\\t $cmp, $s1, $s2, vaddsign0"
VGE_16_vaddSign1|II_VGE_16_vaddSign1|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vge.16\\t $cmp, $s1, $s2, vaddsign1"
VGE_32_vaddSign0|II_VGE_32_vaddSign0|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vge.32\\t $cmp, $s1, $s2, vaddsign0"
VGE_32_vaddSign1|II_VGE_32_vaddSign1|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vge.32\\t $cmp, $s1, $s2, vaddsign1"
VGE_8_vaddSign0|II_VGE_8_vaddSign0|0||cmp:eL|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vge.8\\t $cmp, $s1, $s2, vaddsign0"
VGE_8_vaddSign1|II_VGE_8_vaddSign1|0||cmp:eL|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vge.8\\t $cmp, $s1, $s2, vaddsign1"
VGE_bf16|II_VGE_bf16|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||||"vge.bf16\\t $cmp, $s1, $s2"
VINSERT_16_mIdxImm0|II_VINSERT_16_mIdxImm0|0||dst:OP_mXm|s1:OP_mXm,src:eR||||"vinsert.16\\t $dst, $s1, #0, $src"
VINSERT_16_mR29_insert|II_VINSERT_16_mR29_insert|0||dst:OP_mXm|s1:OP_mXm,idx:mR29_insert,src:eR||||"vinsert.16\\t $dst, $s1, r29, $src"
VINSERT_32_mIdxImm0|II_VINSERT_32_mIdxImm0|0||dst:OP_mXm|s1:OP_mXm,src:eR||||"vinsert.32\\t $dst, $s1, #0, $src"
VINSERT_32_mR29_insert|II_VINSERT_32_mR29_insert|0||dst:OP_mXm|s1:OP_mXm,idx:mR29_insert,src:eR||||"vinsert.32\\t $dst, $s1, r29, $src"
VINSERT_64_mIdxImm0|II_VINSERT_64_mIdxImm0|0||dst:OP_mXm|s1:OP_mXm,src:eL||||"vinsert.64\\t $dst, $s1, #0, $src"
VINSERT_64_mR29_insert|II_VINSERT_64_mR29_insert|0||dst:OP_mXm|s1:OP_mXm,idx:mR29_insert,src:eL||||"vinsert.64\\t $dst, $s1, r29, $src"
VINSERT_8_mIdxImm0|II_VINSERT_8_mIdxImm0|0||dst:OP_mXm|s1:OP_mXm,src:eR||||"vinsert.8\\t $dst, $s1, #0, $src"
VINSERT_8_mR29_insert|II_VINSERT_8_mR29_insert|0||dst:OP_mXm|s1:OP_mXm,idx:mR29_insert,src:eR||||"vinsert.8\\t $dst, $s1, r29, $src"
VLDA_128_dmv_lda_w_idx|II_VLDA_128_dmv_lda_w_idx|4000||dst:OP_mWa|ptr:eP,dj:eDJ||||"vlda.128\\t $dst, [$ptr, $dj]"
VLDA_128_dmv_lda_w_idx_imm|II_VLDA_128_dmv_lda_w_idx_imm|4000||dst:OP_mWa|ptr:eP,imm:c8s_step16||||"vlda.128\\t $dst, [$ptr, #$imm]"
VLDA_128_dmv_lda_w_pstm_nrm|II_VLDA_128_dmv_lda_w_pstm_nrm|4000||dst:OP_mWa,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vlda.128\\t $dst, [$ptr], $mod"
VLDA_128_dmv_lda_w_pstm_nrm_imm|II_VLDA_128_dmv_lda_w_pstm_nrm_imm|4000||dst:OP_mWa,ptr_out:eP|ptr:eP,imm:c8s_step16|||ptr_out=ptr|"vlda.128\\t $dst, [$ptr], #$imm"
VLDA_128_dmv_lda_w_spill|II_VLDA_128_dmv_lda_w_spill|4000||dst:OP_mWa|imm:c14n_step16||sp||"vlda.128\\t $dst, [sp, #$imm]"
VLDA_2D_128|II_VLDA_2D_128|4006||dst:OP_mWa,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vlda.2d.128\\t $dst, [$ptr], $mod"
VLDA_2D_CONV_fp32_bf16_dmw_lda_ups_bf|II_VLDA_2D_CONV_fp32_bf16_dmw_lda_ups_bf|4006||op:OP_mBMm,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vlda.2d.conv.fp32.bf16\\t $op, [$ptr], $mod"
VLDA_2D_CONV_fp32_bf16_dmx_lda_ups_bf|II_VLDA_2D_CONV_fp32_bf16_dmx_lda_ups_bf|4006||op:OP_mCMm,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vlda.2d.conv.fp32.bf16\\t $op, [$ptr], $mod"
VLDA_2D_UPS_2x_dmw_lda_ups_w2b_upsSign0|II_VLDA_2D_UPS_2x_dmw_lda_ups_w2b_upsSign0|4006||dst:OP_mBMm,ptr_out:eP,dc:eDC|su:eS,ptr:eP,mod:eD|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.2d.ups.2x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_2D_UPS_2x_dmw_lda_ups_w2b_upsSign1|II_VLDA_2D_UPS_2x_dmw_lda_ups_w2b_upsSign1|4006||dst:OP_mBMm,ptr_out:eP,dc:eDC|su:eS,ptr:eP,mod:eD|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.2d.ups.2x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_2D_UPS_2x_dmx_lda_ups_x2c_upsSign0|II_VLDA_2D_UPS_2x_dmx_lda_ups_x2c_upsSign0|4006||dst:OP_mCMm,ptr_out:eP,dc:eDC|su:eS,ptr:eP,mod:eD|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.2d.ups.2x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_2D_UPS_2x_dmx_lda_ups_x2c_upsSign1|II_VLDA_2D_UPS_2x_dmx_lda_ups_x2c_upsSign1|4006||dst:OP_mCMm,ptr_out:eP,dc:eDC|su:eS,ptr:eP,mod:eD|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.2d.ups.2x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_2D_UPS_4x_dmw_lda_ups_w2c_upsSign0|II_VLDA_2D_UPS_4x_dmw_lda_ups_w2c_upsSign0|4006||dst:OP_mCMm,ptr_out:eP,dc:eDC|su:eS,ptr:eP,mod:eD|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.2d.ups.4x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_2D_UPS_4x_dmw_lda_ups_w2c_upsSign1|II_VLDA_2D_UPS_4x_dmw_lda_ups_w2c_upsSign1|4006||dst:OP_mCMm,ptr_out:eP,dc:eDC|su:eS,ptr:eP,mod:eD|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.2d.ups.4x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_2D_UPS_4x_dmx_lda_ups_x2d_upsSign0|II_VLDA_2D_UPS_4x_dmx_lda_ups_x2d_upsSign0|4006||dst:eDM,ptr_out:eP,dc:eDC|su:eS,ptr:eP,mod:eD|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.2d.ups.4x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_2D_UPS_4x_dmx_lda_ups_x2d_upsSign1|II_VLDA_2D_UPS_4x_dmx_lda_ups_x2d_upsSign1|4006||dst:eDM,ptr_out:eP,dc:eDC|su:eS,ptr:eP,mod:eD|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.2d.ups.4x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_2D_dmw_lda_w|II_VLDA_2D_dmw_lda_w|4006||dst:OP_mWa,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vlda.2d\\t $dst, [$ptr], $mod"
VLDA_2D_dmx_lda_bm|II_VLDA_2D_dmx_lda_bm|4006||dst:OP_mBMm,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vlda.2d\\t $dst, [$ptr], $mod"
VLDA_2D_dmx_lda_fifohl|II_VLDA_2D_dmx_lda_fifohl|4006||dst:OP_mFifoHLReg,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vlda.2d\\t $dst, [$ptr], $mod"
VLDA_2D_dmx_lda_x|II_VLDA_2D_dmx_lda_x|4006||dst:OP_mXa,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vlda.2d\\t $dst, [$ptr], $mod"
VLDA_3D_128|II_VLDA_3D_128|4006||dst:OP_mWa,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vlda.3d.128\\t $dst, [$ptr], $mod"
VLDA_3D_CONV_fp32_bf16_dmw_lda_ups_bf|II_VLDA_3D_CONV_fp32_bf16_dmw_lda_ups_bf|4006||op:OP_mBMm,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vlda.3d.conv.fp32.bf16\\t $op, [$ptr], $mod"
VLDA_3D_CONV_fp32_bf16_dmx_lda_ups_bf|II_VLDA_3D_CONV_fp32_bf16_dmx_lda_ups_bf|4006||op:OP_mCMm,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vlda.3d.conv.fp32.bf16\\t $op, [$ptr], $mod"
VLDA_3D_UPS_2x_dmw_lda_ups_w2b_upsSign0|II_VLDA_3D_UPS_2x_dmw_lda_ups_w2b_upsSign0|4006||dst:OP_mBMm,ptr_out:eP,dcl:eDCL,dch:eDCH|su:eS,ptr:eP,mod:eDS|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.3d.ups.2x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_3D_UPS_2x_dmw_lda_ups_w2b_upsSign1|II_VLDA_3D_UPS_2x_dmw_lda_ups_w2b_upsSign1|4006||dst:OP_mBMm,ptr_out:eP,dcl:eDCL,dch:eDCH|su:eS,ptr:eP,mod:eDS|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.3d.ups.2x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_3D_UPS_2x_dmx_lda_ups_x2c_upsSign0|II_VLDA_3D_UPS_2x_dmx_lda_ups_x2c_upsSign0|4006||dst:OP_mCMm,ptr_out:eP,dcl:eDCL,dch:eDCH|su:eS,ptr:eP,mod:eDS|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.3d.ups.2x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_3D_UPS_2x_dmx_lda_ups_x2c_upsSign1|II_VLDA_3D_UPS_2x_dmx_lda_ups_x2c_upsSign1|4006||dst:OP_mCMm,ptr_out:eP,dcl:eDCL,dch:eDCH|su:eS,ptr:eP,mod:eDS|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.3d.ups.2x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_3D_UPS_4x_dmw_lda_ups_w2c_upsSign0|II_VLDA_3D_UPS_4x_dmw_lda_ups_w2c_upsSign0|4006||dst:OP_mCMm,ptr_out:eP,dcl:eDCL,dch:eDCH|su:eS,ptr:eP,mod:eDS|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.3d.ups.4x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_3D_UPS_4x_dmw_lda_ups_w2c_upsSign1|II_VLDA_3D_UPS_4x_dmw_lda_ups_w2c_upsSign1|4006||dst:OP_mCMm,ptr_out:eP,dcl:eDCL,dch:eDCH|su:eS,ptr:eP,mod:eDS|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.3d.ups.4x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_3D_UPS_4x_dmx_lda_ups_x2d_upsSign0|II_VLDA_3D_UPS_4x_dmx_lda_ups_x2d_upsSign0|4006||dst:eDM,ptr_out:eP,dcl:eDCL,dch:eDCH|su:eS,ptr:eP,mod:eDS|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.3d.ups.4x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_3D_UPS_4x_dmx_lda_ups_x2d_upsSign1|II_VLDA_3D_UPS_4x_dmx_lda_ups_x2d_upsSign1|4006||dst:eDM,ptr_out:eP,dcl:eDCL,dch:eDCH|su:eS,ptr:eP,mod:eDS|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.3d.ups.4x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_3D_dmw_lda_w|II_VLDA_3D_dmw_lda_w|4006||dst:OP_mWa,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vlda.3d\\t $dst, [$ptr], $mod"
VLDA_3D_dmx_lda_bm|II_VLDA_3D_dmx_lda_bm|4006||dst:OP_mBMm,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vlda.3d\\t $dst, [$ptr], $mod"
VLDA_3D_dmx_lda_fifohl|II_VLDA_3D_dmx_lda_fifohl|4006||dst:OP_mFifoHLReg,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vlda.3d\\t $dst, [$ptr], $mod"
VLDA_3D_dmx_lda_x|II_VLDA_3D_dmx_lda_x|4006||dst:OP_mXa,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vlda.3d\\t $dst, [$ptr], $mod"
VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_idx|II_VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_idx|4000||op:OP_mBMm|ptr:eP,dj:eDJ||||"vlda.conv.fp32.bf16\\t $op, [$ptr, $dj]"
VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_idx_imm|II_VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_idx_imm|4000||op:OP_mBMm|ptr:eP,imm:c9s_step32||||"vlda.conv.fp32.bf16\\t $op, [$ptr, #$imm]"
VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_pstm_nrm|II_VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_pstm_nrm|4000||op:OP_mBMm,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vlda.conv.fp32.bf16\\t $op, [$ptr], $mod"
VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_pstm_nrm_imm|II_VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_pstm_nrm_imm|4000||op:OP_mBMm,ptr_out:eP|ptr:eP,imm:c9s_step32|||ptr_out=ptr|"vlda.conv.fp32.bf16\\t $op, [$ptr], #$imm"
VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_idx|II_VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_idx|4000||op:OP_mCMm|ptr:eP,dj:eDJ||||"vlda.conv.fp32.bf16\\t $op, [$ptr, $dj]"
VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_idx_imm|II_VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_idx_imm|4000||op:OP_mCMm|ptr:eP,imm:c10s_step64||||"vlda.conv.fp32.bf16\\t $op, [$ptr, #$imm]"
VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_pstm_nrm|II_VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_pstm_nrm|4000||op:OP_mCMm,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vlda.conv.fp32.bf16\\t $op, [$ptr], $mod"
VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_pstm_nrm_imm|II_VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_pstm_nrm_imm|4000||op:OP_mCMm,ptr_out:eP|ptr:eP,imm:c10s_step64|||ptr_out=ptr|"vlda.conv.fp32.bf16\\t $op, [$ptr], #$imm"
VLDA_FILL_512|II_VLDA_FILL_512|4006||ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2||||"vlda.fill.512\\t [$ptr, $fifo_reg, $pos]"
VLDA_POP_512_2D|II_VLDA_POP_512_2D|4006||dst:OP_mXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vlda.pop.512.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_512_3D|II_VLDA_POP_512_3D|4006||dst:OP_mXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vlda.pop.512.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_512_fifo_1d_pop|II_VLDA_POP_512_fifo_1d_pop|4006||dst:OP_mXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vlda.pop.512\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_512_normal_pop|II_VLDA_POP_512_normal_pop|4006||dst:OP_mXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vlda.pop.512\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDA_POP_544_2D|II_VLDA_POP_544_2D|4006||dst:OP_mEXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vlda.pop.544.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_544_3D|II_VLDA_POP_544_3D|4006||dst:OP_mEXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vlda.pop.544.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_544_fifo_1d_pop|II_VLDA_POP_544_fifo_1d_pop|4006||dst:OP_mEXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vlda.pop.544\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_544_normal_pop|II_VLDA_POP_544_normal_pop|4006||dst:OP_mEXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vlda.pop.544\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDA_POP_576_2D|II_VLDA_POP_576_2D|4006||dst:OP_mEXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vlda.pop.576.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_576_3D|II_VLDA_POP_576_3D|4006||dst:OP_mEXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vlda.pop.576.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_576_fifo_1d_pop|II_VLDA_POP_576_fifo_1d_pop|4006||dst:OP_mEXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vlda.pop.576\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_576_normal_pop|II_VLDA_POP_576_normal_pop|4006||dst:OP_mEXa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vlda.pop.576\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDA_POP_640_2D|II_VLDA_POP_640_2D|4006||dst:OP_mQXsa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vlda.pop.640.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_640_3D|II_VLDA_POP_640_3D|4006||dst:OP_mQXsa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vlda.pop.640.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_640_fifo_1d_pop|II_VLDA_POP_640_fifo_1d_pop|4006||dst:OP_mQXsa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vlda.pop.640\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_640_normal_pop|II_VLDA_POP_640_normal_pop|4006||dst:OP_mQXsa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vlda.pop.640\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDA_POP_704_2D|II_VLDA_POP_704_2D|4006||dst:OP_mQEXsa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vlda.pop.704.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_704_3D|II_VLDA_POP_704_3D|4006||dst:OP_mQEXsa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vlda.pop.704.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_704_fifo_1d_pop|II_VLDA_POP_704_fifo_1d_pop|4006||dst:OP_mQEXsa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vlda.pop.704\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDA_POP_704_normal_pop|II_VLDA_POP_704_normal_pop|4006||dst:OP_mQEXsa,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vlda.pop.704\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDA_UPS_2x_dmw_lda_ups_w2b_idx_imm_upsSign0|II_VLDA_UPS_2x_dmw_lda_ups_w2b_idx_imm_upsSign0|4000||dst:OP_mBMm|su:eS,ptr:eP,imm:c9s_step32|srUPS_of|crSat,crUPSMode,upsSign0||"vlda.ups.2x\\t $dst, $su, upssign0, [$ptr, #$imm]"
VLDA_UPS_2x_dmw_lda_ups_w2b_idx_imm_upsSign1|II_VLDA_UPS_2x_dmw_lda_ups_w2b_idx_imm_upsSign1|4000||dst:OP_mBMm|su:eS,ptr:eP,imm:c9s_step32|srUPS_of|crSat,crUPSMode,upsSign1||"vlda.ups.2x\\t $dst, $su, upssign1, [$ptr, #$imm]"
VLDA_UPS_2x_dmw_lda_ups_w2b_idx_upsSign0|II_VLDA_UPS_2x_dmw_lda_ups_w2b_idx_upsSign0|4000||dst:OP_mBMm|su:eS,ptr:eP,dj:eDJ|srUPS_of|crSat,crUPSMode,upsSign0||"vlda.ups.2x\\t $dst, $su, upssign0, [$ptr, $dj]"
VLDA_UPS_2x_dmw_lda_ups_w2b_idx_upsSign1|II_VLDA_UPS_2x_dmw_lda_ups_w2b_idx_upsSign1|4000||dst:OP_mBMm|su:eS,ptr:eP,dj:eDJ|srUPS_of|crSat,crUPSMode,upsSign1||"vlda.ups.2x\\t $dst, $su, upssign1, [$ptr, $dj]"
VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_imm_upsSign0|II_VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_imm_upsSign0|4000||dst:OP_mBMm,ptr_out:eP|su:eS,ptr:eP,imm:c9s_step32|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.ups.2x\\t $dst, $su, upssign0, [$ptr], #$imm"
VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_imm_upsSign1|II_VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_imm_upsSign1|4000||dst:OP_mBMm,ptr_out:eP|su:eS,ptr:eP,imm:c9s_step32|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.ups.2x\\t $dst, $su, upssign1, [$ptr], #$imm"
VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_upsSign0|II_VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_upsSign0|4000||dst:OP_mBMm,ptr_out:eP|su:eS,ptr:eP,mod:eM|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.ups.2x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_upsSign1|II_VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_upsSign1|4000||dst:OP_mBMm,ptr_out:eP|su:eS,ptr:eP,mod:eM|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.ups.2x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_UPS_2x_dmx_lda_ups_x2c_idx_imm_upsSign0|II_VLDA_UPS_2x_dmx_lda_ups_x2c_idx_imm_upsSign0|4000||dst:OP_mCMm|su:eS,ptr:eP,imm:c10s_step64|srUPS_of|crSat,crUPSMode,upsSign0||"vlda.ups.2x\\t $dst, $su, upssign0, [$ptr, #$imm]"
VLDA_UPS_2x_dmx_lda_ups_x2c_idx_imm_upsSign1|II_VLDA_UPS_2x_dmx_lda_ups_x2c_idx_imm_upsSign1|4000||dst:OP_mCMm|su:eS,ptr:eP,imm:c10s_step64|srUPS_of|crSat,crUPSMode,upsSign1||"vlda.ups.2x\\t $dst, $su, upssign1, [$ptr, #$imm]"
VLDA_UPS_2x_dmx_lda_ups_x2c_idx_upsSign0|II_VLDA_UPS_2x_dmx_lda_ups_x2c_idx_upsSign0|4000||dst:OP_mCMm|su:eS,ptr:eP,dj:eDJ|srUPS_of|crSat,crUPSMode,upsSign0||"vlda.ups.2x\\t $dst, $su, upssign0, [$ptr, $dj]"
VLDA_UPS_2x_dmx_lda_ups_x2c_idx_upsSign1|II_VLDA_UPS_2x_dmx_lda_ups_x2c_idx_upsSign1|4000||dst:OP_mCMm|su:eS,ptr:eP,dj:eDJ|srUPS_of|crSat,crUPSMode,upsSign1||"vlda.ups.2x\\t $dst, $su, upssign1, [$ptr, $dj]"
VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_imm_upsSign0|II_VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_imm_upsSign0|4000||dst:OP_mCMm,ptr_out:eP|su:eS,ptr:eP,imm:c10s_step64|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.ups.2x\\t $dst, $su, upssign0, [$ptr], #$imm"
VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_imm_upsSign1|II_VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_imm_upsSign1|4000||dst:OP_mCMm,ptr_out:eP|su:eS,ptr:eP,imm:c10s_step64|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.ups.2x\\t $dst, $su, upssign1, [$ptr], #$imm"
VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_upsSign0|II_VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_upsSign0|4000||dst:OP_mCMm,ptr_out:eP|su:eS,ptr:eP,mod:eM|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.ups.2x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_upsSign1|II_VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_upsSign1|4000||dst:OP_mCMm,ptr_out:eP|su:eS,ptr:eP,mod:eM|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.ups.2x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_UPS_4x_dmw_lda_ups_w2c_idx_imm_upsSign0|II_VLDA_UPS_4x_dmw_lda_ups_w2c_idx_imm_upsSign0|4000||dst:OP_mCMm|su:eS,ptr:eP,imm:c9s_step32|srUPS_of|crSat,crUPSMode,upsSign0||"vlda.ups.4x\\t $dst, $su, upssign0, [$ptr, #$imm]"
VLDA_UPS_4x_dmw_lda_ups_w2c_idx_imm_upsSign1|II_VLDA_UPS_4x_dmw_lda_ups_w2c_idx_imm_upsSign1|4000||dst:OP_mCMm|su:eS,ptr:eP,imm:c9s_step32|srUPS_of|crSat,crUPSMode,upsSign1||"vlda.ups.4x\\t $dst, $su, upssign1, [$ptr, #$imm]"
VLDA_UPS_4x_dmw_lda_ups_w2c_idx_upsSign0|II_VLDA_UPS_4x_dmw_lda_ups_w2c_idx_upsSign0|4000||dst:OP_mCMm|su:eS,ptr:eP,dj:eDJ|srUPS_of|crSat,crUPSMode,upsSign0||"vlda.ups.4x\\t $dst, $su, upssign0, [$ptr, $dj]"
VLDA_UPS_4x_dmw_lda_ups_w2c_idx_upsSign1|II_VLDA_UPS_4x_dmw_lda_ups_w2c_idx_upsSign1|4000||dst:OP_mCMm|su:eS,ptr:eP,dj:eDJ|srUPS_of|crSat,crUPSMode,upsSign1||"vlda.ups.4x\\t $dst, $su, upssign1, [$ptr, $dj]"
VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_imm_upsSign0|II_VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_imm_upsSign0|4000||dst:OP_mCMm,ptr_out:eP|su:eS,ptr:eP,imm:c9s_step32|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.ups.4x\\t $dst, $su, upssign0, [$ptr], #$imm"
VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_imm_upsSign1|II_VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_imm_upsSign1|4000||dst:OP_mCMm,ptr_out:eP|su:eS,ptr:eP,imm:c9s_step32|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.ups.4x\\t $dst, $su, upssign1, [$ptr], #$imm"
VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_upsSign0|II_VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_upsSign0|4000||dst:OP_mCMm,ptr_out:eP|su:eS,ptr:eP,mod:eM|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.ups.4x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_upsSign1|II_VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_upsSign1|4000||dst:OP_mCMm,ptr_out:eP|su:eS,ptr:eP,mod:eM|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.ups.4x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_UPS_4x_dmx_lda_ups_x2d_idx_imm_upsSign0|II_VLDA_UPS_4x_dmx_lda_ups_x2d_idx_imm_upsSign0|4000||dst:eDM|su:eS,ptr:eP,imm:c10s_step64|srUPS_of|crSat,crUPSMode,upsSign0||"vlda.ups.4x\\t $dst, $su, upssign0, [$ptr, #$imm]"
VLDA_UPS_4x_dmx_lda_ups_x2d_idx_imm_upsSign1|II_VLDA_UPS_4x_dmx_lda_ups_x2d_idx_imm_upsSign1|4000||dst:eDM|su:eS,ptr:eP,imm:c10s_step64|srUPS_of|crSat,crUPSMode,upsSign1||"vlda.ups.4x\\t $dst, $su, upssign1, [$ptr, #$imm]"
VLDA_UPS_4x_dmx_lda_ups_x2d_idx_upsSign0|II_VLDA_UPS_4x_dmx_lda_ups_x2d_idx_upsSign0|4000||dst:eDM|su:eS,ptr:eP,dj:eDJ|srUPS_of|crSat,crUPSMode,upsSign0||"vlda.ups.4x\\t $dst, $su, upssign0, [$ptr, $dj]"
VLDA_UPS_4x_dmx_lda_ups_x2d_idx_upsSign1|II_VLDA_UPS_4x_dmx_lda_ups_x2d_idx_upsSign1|4000||dst:eDM|su:eS,ptr:eP,dj:eDJ|srUPS_of|crSat,crUPSMode,upsSign1||"vlda.ups.4x\\t $dst, $su, upssign1, [$ptr, $dj]"
VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_imm_upsSign0|II_VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_imm_upsSign0|4000||dst:eDM,ptr_out:eP|su:eS,ptr:eP,imm:c10s_step64|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.ups.4x\\t $dst, $su, upssign0, [$ptr], #$imm"
VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_imm_upsSign1|II_VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_imm_upsSign1|4000||dst:eDM,ptr_out:eP|su:eS,ptr:eP,imm:c10s_step64|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.ups.4x\\t $dst, $su, upssign1, [$ptr], #$imm"
VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_upsSign0|II_VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_upsSign0|4000||dst:eDM,ptr_out:eP|su:eS,ptr:eP,mod:eM|srUPS_of|crSat,crUPSMode,upsSign0|ptr_out=ptr|"vlda.ups.4x\\t $dst, $su, upssign0, [$ptr], $mod"
VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_upsSign1|II_VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_upsSign1|4000||dst:eDM,ptr_out:eP|su:eS,ptr:eP,mod:eM|srUPS_of|crSat,crUPSMode,upsSign1|ptr_out=ptr|"vlda.ups.4x\\t $dst, $su, upssign1, [$ptr], $mod"
VLDA_dmw_lda_w_idx|II_VLDA_dmw_lda_w_idx|4000||dst:OP_mWa|ptr:eP,dj:eDJ||||"vlda\\t $dst, [$ptr, $dj]"
VLDA_dmw_lda_w_idx_imm|II_VLDA_dmw_lda_w_idx_imm|4000||dst:OP_mWa|ptr:eP,imm:c9s_step32||||"vlda\\t $dst, [$ptr, #$imm]"
VLDA_dmw_lda_w_pstm_nrm|II_VLDA_dmw_lda_w_pstm_nrm|4000||dst:OP_mWa,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vlda\\t $dst, [$ptr], $mod"
VLDA_dmw_lda_w_pstm_nrm_imm|II_VLDA_dmw_lda_w_pstm_nrm_imm|4000||dst:OP_mWa,ptr_out:eP|ptr:eP,imm:c9s_step32|||ptr_out=ptr|"vlda\\t $dst, [$ptr], #$imm"
VLDA_dmw_lda_w_spill|II_VLDA_dmw_lda_w_spill|4000||dst:OP_mWa|imm:c15n_step32||sp||"vlda\\t $dst, [sp, #$imm]"
VLDA_dmx_lda_bm_idx|II_VLDA_dmx_lda_bm_idx|4000||dst:OP_mBMm|ptr:eP,dj:eDJ||||"vlda\\t $dst, [$ptr, $dj]"
VLDA_dmx_lda_bm_idx_imm|II_VLDA_dmx_lda_bm_idx_imm|4000||dst:OP_mBMm|ptr:eP,imm:c10s_step64||||"vlda\\t $dst, [$ptr, #$imm]"
VLDA_dmx_lda_bm_pstm_nrm|II_VLDA_dmx_lda_bm_pstm_nrm|4000||dst:OP_mBMm,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vlda\\t $dst, [$ptr], $mod"
VLDA_dmx_lda_bm_pstm_nrm_imm|II_VLDA_dmx_lda_bm_pstm_nrm_imm|4000||dst:OP_mBMm,ptr_out:eP|ptr:eP,imm:c10s_step64|||ptr_out=ptr|"vlda\\t $dst, [$ptr], #$imm"
VLDA_dmx_lda_bm_spill|II_VLDA_dmx_lda_bm_spill|4000||dst:OP_mBMm|imm:c16n_step64||sp||"vlda\\t $dst, [sp, #$imm]"
VLDA_dmx_lda_fifohl_idx|II_VLDA_dmx_lda_fifohl_idx|4000||dst:OP_mFifoHLReg|ptr:eP,dj:eDJ||||"vlda\\t $dst, [$ptr, $dj]"
VLDA_dmx_lda_fifohl_idx_imm|II_VLDA_dmx_lda_fifohl_idx_imm|4000||dst:OP_mFifoHLReg|ptr:eP,imm:c10s_step64||||"vlda\\t $dst, [$ptr, #$imm]"
VLDA_dmx_lda_fifohl_pstm_nrm|II_VLDA_dmx_lda_fifohl_pstm_nrm|4000||dst:OP_mFifoHLReg,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vlda\\t $dst, [$ptr], $mod"
VLDA_dmx_lda_fifohl_pstm_nrm_imm|II_VLDA_dmx_lda_fifohl_pstm_nrm_imm|4000||dst:OP_mFifoHLReg,ptr_out:eP|ptr:eP,imm:c10s_step64|||ptr_out=ptr|"vlda\\t $dst, [$ptr], #$imm"
VLDA_dmx_lda_fifohl_spill|II_VLDA_dmx_lda_fifohl_spill|4000||dst:OP_mFifoHLReg|imm:c16n_step64||sp||"vlda\\t $dst, [sp, #$imm]"
VLDA_dmx_lda_x_idx|II_VLDA_dmx_lda_x_idx|4000||dst:OP_mXa|ptr:eP,dj:eDJ||||"vlda\\t $dst, [$ptr, $dj]"
VLDA_dmx_lda_x_idx_imm|II_VLDA_dmx_lda_x_idx_imm|4000||dst:OP_mXa|ptr:eP,imm:c10s_step64||||"vlda\\t $dst, [$ptr, #$imm]"
VLDA_dmx_lda_x_pstm_nrm|II_VLDA_dmx_lda_x_pstm_nrm|4000||dst:OP_mXa,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vlda\\t $dst, [$ptr], $mod"
VLDA_dmx_lda_x_pstm_nrm_imm|II_VLDA_dmx_lda_x_pstm_nrm_imm|4000||dst:OP_mXa,ptr_out:eP|ptr:eP,imm:c10s_step64|||ptr_out=ptr|"vlda\\t $dst, [$ptr], #$imm"
VLDA_dmx_lda_x_spill|II_VLDA_dmx_lda_x_spill|4000||dst:OP_mXa|imm:c16n_step64||sp||"vlda\\t $dst, [sp, #$imm]"
VLDB_128_idx|II_VLDB_128_idx|4000||dst:OP_mWb|ptr:eP,dj:eDJ||||"vldb.128\\t $dst, [$ptr, $dj]"
VLDB_128_idx_imm|II_VLDB_128_idx_imm|4000||dst:OP_mWb|ptr:eP,imm:c8s_step16||||"vldb.128\\t $dst, [$ptr, #$imm]"
VLDB_128_pstm_nrm|II_VLDB_128_pstm_nrm|4000||dst:OP_mWb,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vldb.128\\t $dst, [$ptr], $mod"
VLDB_128_pstm_nrm_imm|II_VLDB_128_pstm_nrm_imm|4000||dst:OP_mWb,ptr_out:eP|ptr:eP,imm:c8s_step16|||ptr_out=ptr|"vldb.128\\t $dst, [$ptr], #$imm"
VLDB_2D_128|II_VLDB_2D_128|4006||dst:OP_mWb,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vldb.2d.128\\t $dst, [$ptr], $mod"
VLDB_2D_UNPACK_dmw_ldb_unpack_unpackSign0|II_VLDB_2D_UNPACK_dmw_ldb_unpack_unpackSign0|4006||dst:OP_mXb,ptr_out:eP,dc:eDC|ptr:eP,mod:eD||crUnpackSize,unpackSign0|ptr_out=ptr|"vldb.2d.unpack\\t $dst, unpacksign0, [$ptr], $mod"
VLDB_2D_UNPACK_dmw_ldb_unpack_unpackSign1|II_VLDB_2D_UNPACK_dmw_ldb_unpack_unpackSign1|4006||dst:OP_mXb,ptr_out:eP,dc:eDC|ptr:eP,mod:eD||crUnpackSize,unpackSign1|ptr_out=ptr|"vldb.2d.unpack\\t $dst, unpacksign1, [$ptr], $mod"
VLDB_2D_UNPACK_dmx_ldb_unpack_unpackSign0|II_VLDB_2D_UNPACK_dmx_ldb_unpack_unpackSign0|4006||dst:eY,ptr_out:eP,dc:eDC|ptr:eP,mod:eD||crUnpackSize,unpackSign0|ptr_out=ptr|"vldb.2d.unpack\\t $dst, unpacksign0, [$ptr], $mod"
VLDB_2D_UNPACK_dmx_ldb_unpack_unpackSign1|II_VLDB_2D_UNPACK_dmx_ldb_unpack_unpackSign1|4006||dst:eY,ptr_out:eP,dc:eDC|ptr:eP,mod:eD||crUnpackSize,unpackSign1|ptr_out=ptr|"vldb.2d.unpack\\t $dst, unpacksign1, [$ptr], $mod"
VLDB_2D_dmw_ldb|II_VLDB_2D_dmw_ldb|4006||dst:OP_mWb,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vldb.2d\\t $dst, [$ptr], $mod"
VLDB_2D_dmx_ldb_x|II_VLDB_2D_dmx_ldb_x|4006||dst:OP_mXb,ptr_out:eP,dc:eDC|ptr:eP,mod:eD|||ptr_out=ptr|"vldb.2d\\t $dst, [$ptr], $mod"
VLDB_3D_128|II_VLDB_3D_128|4006||dst:OP_mWb,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vldb.3d.128\\t $dst, [$ptr], $mod"
VLDB_3D_UNPACK_dmw_ldb_unpack_unpackSign0|II_VLDB_3D_UNPACK_dmw_ldb_unpack_unpackSign0|4006||dst:OP_mXb,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS||crUnpackSize,unpackSign0|ptr_out=ptr|"vldb.3d.unpack\\t $dst, unpacksign0, [$ptr], $mod"
VLDB_3D_UNPACK_dmw_ldb_unpack_unpackSign1|II_VLDB_3D_UNPACK_dmw_ldb_unpack_unpackSign1|4006||dst:OP_mXb,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS||crUnpackSize,unpackSign1|ptr_out=ptr|"vldb.3d.unpack\\t $dst, unpacksign1, [$ptr], $mod"
VLDB_3D_UNPACK_dmx_ldb_unpack_unpackSign0|II_VLDB_3D_UNPACK_dmx_ldb_unpack_unpackSign0|4006||dst:eY,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS||crUnpackSize,unpackSign0|ptr_out=ptr|"vldb.3d.unpack\\t $dst, unpacksign0, [$ptr], $mod"
VLDB_3D_UNPACK_dmx_ldb_unpack_unpackSign1|II_VLDB_3D_UNPACK_dmx_ldb_unpack_unpackSign1|4006||dst:eY,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS||crUnpackSize,unpackSign1|ptr_out=ptr|"vldb.3d.unpack\\t $dst, unpacksign1, [$ptr], $mod"
VLDB_3D_dmw_ldb|II_VLDB_3D_dmw_ldb|4006||dst:OP_mWb,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vldb.3d\\t $dst, [$ptr], $mod"
VLDB_3D_dmx_ldb_x|II_VLDB_3D_dmx_ldb_x|4006||dst:OP_mXb,ptr_out:eP,dcl:eDCL,dch:eDCH|ptr:eP,mod:eDS|||ptr_out=ptr|"vldb.3d\\t $dst, [$ptr], $mod"
VLDB_4x16_hi|II_VLDB_4x16_hi|4000||dst:OP_mWb|src:OP_mWs||||"vldb.4x16.hi\\t $dst, $src"
VLDB_4x16_lo|II_VLDB_4x16_lo|4000||dst:OP_mWb|src:OP_mWs||||"vldb.4x16.lo\\t $dst, $src"
VLDB_4x32_hi|II_VLDB_4x32_hi|4000||dst:OP_mWb|src:OP_mWs||||"vldb.4x32.hi\\t $dst, $src"
VLDB_4x32_lo|II_VLDB_4x32_lo|4000||dst:OP_mWb|src:OP_mWs||||"vldb.4x32.lo\\t $dst, $src"
VLDB_4x64_hi|II_VLDB_4x64_hi|4000||dst:OP_mWb|src:OP_mWs||||"vldb.4x64.hi\\t $dst, $src"
VLDB_4x64_lo|II_VLDB_4x64_lo|4000||dst:OP_mWb|src:OP_mWs||||"vldb.4x64.lo\\t $dst, $src"
VLDB_FILLX_512|II_VLDB_FILLX_512|4006||ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|conf_e1:mR30_fifo_step_e1,conf_e7:mR30_fifo_step_e7,ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|lfe|lfe||"vldb.fillx.512\\t [$ptr, $fifo_reg, $pos]"
VLDB_FILL_512|II_VLDB_FILL_512|4006||ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2||||"vldb.fill.512\\t [$ptr, $fifo_reg, $pos]"
VLDB_POPX_512|II_VLDB_POPX_512|4006||dst:OP_mXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|conf_e1:mR30_fifo_step_e1,conf_e7:mR30_fifo_step_e7,ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|lfe,srFifo_uf|lfe||"vldb.popx.512\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDB_POP_512_2D|II_VLDB_POP_512_2D|4006||dst:OP_mXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vldb.pop.512.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_512_3D|II_VLDB_POP_512_3D|4006||dst:OP_mXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vldb.pop.512.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_512_fifo_1d_pop|II_VLDB_POP_512_fifo_1d_pop|4006||dst:OP_mXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vldb.pop.512\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_512_normal_pop|II_VLDB_POP_512_normal_pop|4006||dst:OP_mXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vldb.pop.512\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDB_POP_544_2D|II_VLDB_POP_544_2D|4006||dst:OP_mEXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vldb.pop.544.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_544_3D|II_VLDB_POP_544_3D|4006||dst:OP_mEXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vldb.pop.544.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_544_fifo_1d_pop|II_VLDB_POP_544_fifo_1d_pop|4006||dst:OP_mEXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vldb.pop.544\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_544_normal_pop|II_VLDB_POP_544_normal_pop|4006||dst:OP_mEXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vldb.pop.544\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDB_POP_576_2D|II_VLDB_POP_576_2D|4006||dst:OP_mEXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vldb.pop.576.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_576_3D|II_VLDB_POP_576_3D|4006||dst:OP_mEXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vldb.pop.576.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_576_fifo_1d_pop|II_VLDB_POP_576_fifo_1d_pop|4006||dst:OP_mEXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vldb.pop.576\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_576_normal_pop|II_VLDB_POP_576_normal_pop|4006||dst:OP_mEXb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vldb.pop.576\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDB_POP_640_2D|II_VLDB_POP_640_2D|4006||dst:OP_mQXsb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vldb.pop.640.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_640_3D|II_VLDB_POP_640_3D|4006||dst:OP_mQXsb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vldb.pop.640.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_640_fifo_1d_pop|II_VLDB_POP_640_fifo_1d_pop|4006||dst:OP_mQXsb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vldb.pop.640\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_640_normal_pop|II_VLDB_POP_640_normal_pop|4006||dst:OP_mQXsb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vldb.pop.640\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDB_POP_704_2D|II_VLDB_POP_704_2D|4006||dst:OP_mQEXsb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dc:eDC|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eD|srFifo_uf|||"vldb.pop.704.2d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_704_3D|II_VLDB_POP_704_3D|4006||dst:OP_mQEXsb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2,dcl:eDCL,dch:eDCH|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eDS|srFifo_uf|||"vldb.pop.704.3d\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_704_fifo_1d_pop|II_VLDB_POP_704_fifo_1d_pop|4006||dst:OP_mQEXsb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2,mod:eM|srFifo_uf|||"vldb.pop.704\\t $dst, [$ptr, $fifo_reg, $pos, $mod]"
VLDB_POP_704_normal_pop|II_VLDB_POP_704_normal_pop|4006||dst:OP_mQEXsb,ptr_out:ePS,fifo_reg_out:eLdFifoReg,pos_out:eRF2|ptr:ePS,fifo_reg:eLdFifoReg,pos:eRF2|srFifo_uf|||"vldb.pop.704\\t $dst, [$ptr, $fifo_reg, $pos]"
VLDB_UNPACK_dmw_ldb_unpack_idx_imm_unpackSign0|II_VLDB_UNPACK_dmw_ldb_unpack_idx_imm_unpackSign0|4000||dst:OP_mXb|ptr:eP,imm:c9s_step32||crUnpackSize,unpackSign0||"vldb.unpack\\t $dst, unpacksign0, [$ptr, #$imm]"
VLDB_UNPACK_dmw_ldb_unpack_idx_imm_unpackSign1|II_VLDB_UNPACK_dmw_ldb_unpack_idx_imm_unpackSign1|4000||dst:OP_mXb|ptr:eP,imm:c9s_step32||crUnpackSize,unpackSign1||"vldb.unpack\\t $dst, unpacksign1, [$ptr, #$imm]"
VLDB_UNPACK_dmw_ldb_unpack_idx_unpackSign0|II_VLDB_UNPACK_dmw_ldb_unpack_idx_unpackSign0|4000||dst:OP_mXb|ptr:eP,dj:eDJ||crUnpackSize,unpackSign0||"vldb.unpack\\t $dst, unpacksign0, [$ptr, $dj]"
VLDB_UNPACK_dmw_ldb_unpack_idx_unpackSign1|II_VLDB_UNPACK_dmw_ldb_unpack_idx_unpackSign1|4000||dst:OP_mXb|ptr:eP,dj:eDJ||crUnpackSize,unpackSign1||"vldb.unpack\\t $dst, unpacksign1, [$ptr, $dj]"
VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_imm_unpackSign0|II_VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_imm_unpackSign0|4000||dst:OP_mXb,ptr_out:eP|ptr:eP,imm:c9s_step32||crUnpackSize,unpackSign0|ptr_out=ptr|"vldb.unpack\\t $dst, unpacksign0, [$ptr], #$imm"
VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_imm_unpackSign1|II_VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_imm_unpackSign1|4000||dst:OP_mXb,ptr_out:eP|ptr:eP,imm:c9s_step32||crUnpackSize,unpackSign1|ptr_out=ptr|"vldb.unpack\\t $dst, unpacksign1, [$ptr], #$imm"
VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_unpackSign0|II_VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_unpackSign0|4000||dst:OP_mXb,ptr_out:eP|ptr:eP,mod:eM||crUnpackSize,unpackSign0|ptr_out=ptr|"vldb.unpack\\t $dst, unpacksign0, [$ptr], $mod"
VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_unpackSign1|II_VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_unpackSign1|4000||dst:OP_mXb,ptr_out:eP|ptr:eP,mod:eM||crUnpackSize,unpackSign1|ptr_out=ptr|"vldb.unpack\\t $dst, unpacksign1, [$ptr], $mod"
VLDB_UNPACK_dmx_ldb_unpack_idx_imm_unpackSign0|II_VLDB_UNPACK_dmx_ldb_unpack_idx_imm_unpackSign0|4000||dst:eY|ptr:eP,imm:c10s_step64||crUnpackSize,unpackSign0||"vldb.unpack\\t $dst, unpacksign0, [$ptr, #$imm]"
VLDB_UNPACK_dmx_ldb_unpack_idx_imm_unpackSign1|II_VLDB_UNPACK_dmx_ldb_unpack_idx_imm_unpackSign1|4000||dst:eY|ptr:eP,imm:c10s_step64||crUnpackSize,unpackSign1||"vldb.unpack\\t $dst, unpacksign1, [$ptr, #$imm]"
VLDB_UNPACK_dmx_ldb_unpack_idx_unpackSign0|II_VLDB_UNPACK_dmx_ldb_unpack_idx_unpackSign0|4000||dst:eY|ptr:eP,dj:eDJ||crUnpackSize,unpackSign0||"vldb.unpack\\t $dst, unpacksign0, [$ptr, $dj]"
VLDB_UNPACK_dmx_ldb_unpack_idx_unpackSign1|II_VLDB_UNPACK_dmx_ldb_unpack_idx_unpackSign1|4000||dst:eY|ptr:eP,dj:eDJ||crUnpackSize,unpackSign1||"vldb.unpack\\t $dst, unpacksign1, [$ptr, $dj]"
VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign0|II_VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign0|4000||dst:eY,ptr_out:eP|ptr:eP,imm:c10s_step64||crUnpackSize,unpackSign0|ptr_out=ptr|"vldb.unpack\\t $dst, unpacksign0, [$ptr], #$imm"
VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1|II_VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1|4000||dst:eY,ptr_out:eP|ptr:eP,imm:c10s_step64||crUnpackSize,unpackSign1|ptr_out=ptr|"vldb.unpack\\t $dst, unpacksign1, [$ptr], #$imm"
VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_unpackSign0|II_VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_unpackSign0|4000||dst:eY,ptr_out:eP|ptr:eP,mod:eM||crUnpackSize,unpackSign0|ptr_out=ptr|"vldb.unpack\\t $dst, unpacksign0, [$ptr], $mod"
VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_unpackSign1|II_VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_unpackSign1|4000||dst:eY,ptr_out:eP|ptr:eP,mod:eM||crUnpackSize,unpackSign1|ptr_out=ptr|"vldb.unpack\\t $dst, unpacksign1, [$ptr], $mod"
VLDB_dmw_ldb_idx|II_VLDB_dmw_ldb_idx|4000||dst:OP_mWb|ptr:eP,dj:eDJ||||"vldb\\t $dst, [$ptr, $dj]"
VLDB_dmw_ldb_idx_imm|II_VLDB_dmw_ldb_idx_imm|4000||dst:OP_mWb|ptr:eP,imm:c9s_step32||||"vldb\\t $dst, [$ptr, #$imm]"
VLDB_dmw_ldb_pstm_nrm|II_VLDB_dmw_ldb_pstm_nrm|4000||dst:OP_mWb,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vldb\\t $dst, [$ptr], $mod"
VLDB_dmw_ldb_pstm_nrm_imm|II_VLDB_dmw_ldb_pstm_nrm_imm|4000||dst:OP_mWb,ptr_out:eP|ptr:eP,imm:c9s_step32|||ptr_out=ptr|"vldb\\t $dst, [$ptr], #$imm"
VLDB_dmx_ldb_x_idx|II_VLDB_dmx_ldb_x_idx|4000||dst:OP_mXb|ptr:eP,dj:eDJ||||"vldb\\t $dst, [$ptr, $dj]"
VLDB_dmx_ldb_x_idx_imm|II_VLDB_dmx_ldb_x_idx_imm|4000||dst:OP_mXb|ptr:eP,imm:c10s_step64||||"vldb\\t $dst, [$ptr, #$imm]"
VLDB_dmx_ldb_x_pstm_nrm|II_VLDB_dmx_ldb_x_pstm_nrm|4000||dst:OP_mXb,ptr_out:eP|ptr:eP,mod:eM|||ptr_out=ptr|"vldb\\t $dst, [$ptr], $mod"
VLDB_dmx_ldb_x_pstm_nrm_imm|II_VLDB_dmx_ldb_x_pstm_nrm_imm|4000||dst:OP_mXb,ptr_out:eP|ptr:eP,imm:c10s_step64|||ptr_out=ptr|"vldb\\t $dst, [$ptr], #$imm"
VLT_16_vaddSign0|II_VLT_16_vaddSign0|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vlt.16\\t $cmp, $s1, $s2, vaddsign0"
VLT_16_vaddSign1|II_VLT_16_vaddSign1|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vlt.16\\t $cmp, $s1, $s2, vaddsign1"
VLT_32_vaddSign0|II_VLT_32_vaddSign0|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vlt.32\\t $cmp, $s1, $s2, vaddsign0"
VLT_32_vaddSign1|II_VLT_32_vaddSign1|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vlt.32\\t $cmp, $s1, $s2, vaddsign1"
VLT_8_vaddSign0|II_VLT_8_vaddSign0|0||cmp:eL|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vlt.8\\t $cmp, $s1, $s2, vaddsign0"
VLT_8_vaddSign1|II_VLT_8_vaddSign1|0||cmp:eL|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vlt.8\\t $cmp, $s1, $s2, vaddsign1"
VLT_bf16|II_VLT_bf16|0||cmp:eRS16|s1:OP_mXm,s2:OP_mXn||||"vlt.bf16\\t $cmp, $s1, $s2"
VMAC_f_vmac_bf_vmul_bf_core_X_X|II_VMAC_f_vmac_bf_vmul_bf_core_X_X|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask||"vmac.f\\t$dst, $acc1, $s1, $s2, $acc"
VMAC_f_vmac_bf_vmul_bf_core_Y_Y|II_VMAC_f_vmac_bf_vmul_bf_core_Y_Y|0||dst:eDM|acc1:eDM,s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask||"vmac.f\\t$dst, $acc1, $s1, $s2, $acc"
VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX|II_VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX|0||dst:eDM|acc1:eDM,s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask||"vmac.f\\t$dst, $acc1, $s1, $s2, $acc"
VMAC_f_vmac_bfp_vmul_bfp_core_EX_EY|II_VMAC_f_vmac_bfp_vmul_bfp_core_EX_EY|0||dst:eDM|acc1:eDM,s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask||"vmac.f\\t$dst, $acc1, $s1, $s2, $acc"
VMAC_f_vmac_bfp_vmul_bfp_core_EX_QEY|II_VMAC_f_vmac_bfp_vmul_bfp_core_EX_QEY|0||dst:eDM|acc1:eDM,s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask||"vmac.f\\t$dst, $acc1, $s1, $s2, $acc"
VMAC_f_vmac_bfp_vmul_bfp_core_EY_QEX|II_VMAC_f_vmac_bfp_vmul_bfp_core_EY_QEX|0||dst:eDM|acc1:eDM,s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask||"vmac.f\\t$dst, $acc1, $s1, $s2, $acc"
VMAC_vmul_cm_core_X_QX|II_VMAC_vmul_cm_core_X_QX|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of|||"vmac\\t$dst, $acc1, $s1, $s2,$acc"
VMAC_vmul_cm_core_X_QY|II_VMAC_vmul_cm_core_X_QY|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of|||"vmac\\t$dst, $acc1, $s1, $s2,$acc"
VMAC_vmul_cm_core_X_X|II_VMAC_vmul_cm_core_X_X|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:OP_mXw,acc:eR||||"vmac\\t$dst, $acc1, $s1, $s2,$acc"
VMAC_vmul_cm_core_X_Y|II_VMAC_vmul_cm_core_X_Y|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:eY,acc:eR||||"vmac\\t$dst, $acc1, $s1, $s2,$acc"
VMAC_vmul_cm_core_Y_QX|II_VMAC_vmul_cm_core_Y_QX|0||dst:eDM|acc1:eDM,s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of|||"vmac\\t$dst, $acc1, $s1, $s2,$acc"
VMAC_vmul_cm_core_Y_QY|II_VMAC_vmul_cm_core_Y_QY|0||dst:eDM|acc1:eDM,s1:eY,s2:eQYs,acc:eR|srSparse_of|||"vmac\\t$dst, $acc1, $s1, $s2,$acc"
VMAC_vmul_cm_core_Y_X|II_VMAC_vmul_cm_core_Y_X|0||dst:eDM|acc1:eDM,s1:eY,s2:OP_mXw,acc:eR||||"vmac\\t$dst, $acc1, $s1, $s2,$acc"
VMAC_vmul_cm_core_Y_Y|II_VMAC_vmul_cm_core_Y_Y|0||dst:eDM|acc1:eDM,s1:eY,s2:eY,acc:eR||||"vmac\\t$dst, $acc1, $s1, $s2,$acc"
VMAXDIFF_LT_16_vaddSign0|II_VMAXDIFF_LT_16_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmaxdiff_lt.16\\t $d, r16, $s1, $s2, vaddsign0"
VMAXDIFF_LT_16_vaddSign1|II_VMAXDIFF_LT_16_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmaxdiff_lt.16\\t $d, r16, $s1, $s2, vaddsign1"
VMAXDIFF_LT_32_vaddSign0|II_VMAXDIFF_LT_32_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmaxdiff_lt.32\\t $d, r16, $s1, $s2, vaddsign0"
VMAXDIFF_LT_32_vaddSign1|II_VMAXDIFF_LT_32_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmaxdiff_lt.32\\t $d, r16, $s1, $s2, vaddsign1"
VMAXDIFF_LT_8_vaddSign0|II_VMAXDIFF_LT_8_vaddSign0|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmaxdiff_lt.8\\t $d, r17:r16, $s1, $s2, vaddsign0"
VMAXDIFF_LT_8_vaddSign1|II_VMAXDIFF_LT_8_vaddSign1|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmaxdiff_lt.8\\t $d, r17:r16, $s1, $s2, vaddsign1"
VMAX_LT_16_vaddSign0|II_VMAX_LT_16_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmax_lt.16\\t $d, r16, $s1, $s2, vaddsign0"
VMAX_LT_16_vaddSign1|II_VMAX_LT_16_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmax_lt.16\\t $d, r16, $s1, $s2, vaddsign1"
VMAX_LT_32_vaddSign0|II_VMAX_LT_32_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmax_lt.32\\t $d, r16, $s1, $s2, vaddsign0"
VMAX_LT_32_vaddSign1|II_VMAX_LT_32_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmax_lt.32\\t $d, r16, $s1, $s2, vaddsign1"
VMAX_LT_8_vaddSign0|II_VMAX_LT_8_vaddSign0|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmax_lt.8\\t $d, r17:r16, $s1, $s2, vaddsign0"
VMAX_LT_8_vaddSign1|II_VMAX_LT_8_vaddSign1|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmax_lt.8\\t $d, r17:r16, $s1, $s2, vaddsign1"
VMAX_LT_bf16|II_VMAX_LT_bf16|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||||"vmax_lt.bf16\\t $d, r16, $s1, $s2"
VMIN_GE_16_vaddSign0|II_VMIN_GE_16_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmin_ge.16\\t $d, r16, $s1, $s2, vaddsign0"
VMIN_GE_16_vaddSign1|II_VMIN_GE_16_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmin_ge.16\\t $d, r16, $s1, $s2, vaddsign1"
VMIN_GE_32_vaddSign0|II_VMIN_GE_32_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmin_ge.32\\t $d, r16, $s1, $s2, vaddsign0"
VMIN_GE_32_vaddSign1|II_VMIN_GE_32_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmin_ge.32\\t $d, r16, $s1, $s2, vaddsign1"
VMIN_GE_8_vaddSign0|II_VMIN_GE_8_vaddSign0|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vmin_ge.8\\t $d, r17:r16, $s1, $s2, vaddsign0"
VMIN_GE_8_vaddSign1|II_VMIN_GE_8_vaddSign1|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vmin_ge.8\\t $d, r17:r16, $s1, $s2, vaddsign1"
VMIN_GE_bf16|II_VMIN_GE_bf16|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||||"vmin_ge.bf16\\t $d, r16, $s1, $s2"
VMOV_0_mv_scd_cm|II_VMOV_0_mv_scd_cm|8||dst:OP_mCMm|||crSCDEn||"vmov.0\\t $dst, scd"
VMOV_0_mv_scd_dm_imm|II_VMOV_0_mv_scd_dm_imm|8||dst:eDM|||crSCDEn||"vmov.0\\t $dst, scd"
VMOV_1_mv_scd_cm|II_VMOV_1_mv_scd_cm|8||dst:OP_mCMm|||crSCDEn||"vmov.1\\t $dst, scd"
VMOV_1_mv_scd_dm_imm|II_VMOV_1_mv_scd_dm_imm|8||dst:eDM|||crSCDEn||"vmov.1\\t $dst, scd"
VMOV_2|II_VMOV_2|8||dst:eDM|||crSCDEn||"vmov.2\\t $dst, scd"
VMOV_3|II_VMOV_3|8||dst:eDM|||crSCDEn||"vmov.3\\t $dst, scd"
VMOV_D|II_VMOV_D|0||dst:eDM|acc1:eDM||||"vmov.d\\t$dst, $acc1"
VMOV_alu_mv_mv_cm|II_VMOV_alu_mv_mv_cm|0||dst:OP_mCMm|src:OP_mCMm||||"vmov\\t$dst, $src"
VMOV_alu_mv_mv_ex|II_VMOV_alu_mv_mv_ex|0||dst:OP_mEXm|src:OP_mEXm||||"vmov\\t$dst, $src"
VMOV_alu_mv_mv_q|II_VMOV_alu_mv_mv_q|0||dst:OP_mQQsm|src:OP_mQQsm||||"vmov\\t$dst, $src"
VMOV_alu_mv_mv_q_to_w|II_VMOV_alu_mv_mv_q_to_w|0||dst:OP_mWm|src:OP_mQQsm||||"vmov\\t$dst, $src"
VMOV_alu_mv_mv_qex|II_VMOV_alu_mv_mv_qex|0||dst:OP_mQEXsm|src:OP_mQEXsm||||"vmov\\t$dst, $src"
VMOV_alu_mv_mv_qx|II_VMOV_alu_mv_mv_qx|0||dst:OP_mQXsm|src:OP_mQXsm||||"vmov\\t$dst, $src"
VMOV_alu_mv_mv_w|II_VMOV_alu_mv_mv_w|0||dst:OP_mWm|src:OP_mWm||||"vmov\\t$dst, $src"
VMOV_alu_mv_mv_w_to_q|II_VMOV_alu_mv_mv_w_to_q|0||dst:OP_mQQsm|src:OP_mWm||||"vmov\\t$dst, $src"
VMOV_alu_mv_mv_x|II_VMOV_alu_mv_mv_x|0||dst:OP_mMvBMXDst|src:OP_mMvBMXSrc||||"vmov\\t$dst, $src"
VMOV_lda_mv_scd_bm|II_VMOV_lda_mv_scd_bm|8||dst:OP_mBMm|||crSCDEn||"vmov\\t$dst, scd"
VMOV_lda_mv_scd_dm_dyn|II_VMOV_lda_mv_scd_dm_dyn|8||dst:eDM,r_out:mR31_scd|r:mR31_scd||crSCDEn|r_out=r|"vmov\\t$dst, scd, r31++"
VMOV_lda_mv_scd_dm_reg|II_VMOV_lda_mv_scd_dm_reg|8||dst:eDM|c:mR31_scd||crSCDEn||"vmov\\t$dst, scd, r31"
VMOV_lda_mv_scd_x|II_VMOV_lda_mv_scd_x|8||dst:OP_mXa|||crSCDEn||"vmov\\t$dst, scd"
VMOV_st_mv_mcd_bm|II_VMOV_st_mv_mcd_bm|8|||src:OP_mBMs||crMCDEn||"vmov\\tmcd, $src"
VMOV_st_mv_mcd_x|II_VMOV_st_mv_mcd_x|8|||src:OP_mXs||crMCDEn||"vmov\\tmcd, $src"
VMSC_f_vmac_bf_vmul_bf_core_X_X|II_VMSC_f_vmac_bf_vmul_bf_core_X_X|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask||"vmsc.f\\t$dst, $acc1, $s1, $s2, $acc"
VMSC_f_vmac_bf_vmul_bf_core_Y_Y|II_VMSC_f_vmac_bf_vmul_bf_core_Y_Y|0||dst:eDM|acc1:eDM,s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask||"vmsc.f\\t$dst, $acc1, $s1, $s2, $acc"
VMSC_f_vmac_bfp_vmul_bfp_core_EX_EX|II_VMSC_f_vmac_bfp_vmul_bfp_core_EX_EX|0||dst:eDM|acc1:eDM,s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask||"vmsc.f\\t$dst, $acc1, $s1, $s2, $acc"
VMSC_f_vmac_bfp_vmul_bfp_core_EX_EY|II_VMSC_f_vmac_bfp_vmul_bfp_core_EX_EY|0||dst:eDM|acc1:eDM,s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask||"vmsc.f\\t$dst, $acc1, $s1, $s2, $acc"
VMSC_f_vmac_bfp_vmul_bfp_core_EX_QEY|II_VMSC_f_vmac_bfp_vmul_bfp_core_EX_QEY|0||dst:eDM|acc1:eDM,s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask||"vmsc.f\\t$dst, $acc1, $s1, $s2, $acc"
VMSC_f_vmac_bfp_vmul_bfp_core_EY_QEX|II_VMSC_f_vmac_bfp_vmul_bfp_core_EY_QEX|0||dst:eDM|acc1:eDM,s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask||"vmsc.f\\t$dst, $acc1, $s1, $s2, $acc"
VMSC_vmul_cm_core_X_QX|II_VMSC_vmul_cm_core_X_QX|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of|||"vmsc\\t$dst, $acc1, $s1, $s2,$acc"
VMSC_vmul_cm_core_X_QY|II_VMSC_vmul_cm_core_X_QY|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of|||"vmsc\\t$dst, $acc1, $s1, $s2,$acc"
VMSC_vmul_cm_core_X_X|II_VMSC_vmul_cm_core_X_X|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:OP_mXw,acc:eR||||"vmsc\\t$dst, $acc1, $s1, $s2,$acc"
VMSC_vmul_cm_core_X_Y|II_VMSC_vmul_cm_core_X_Y|0||dst:eDM|acc1:eDM,s1:OP_mXv,s2:eY,acc:eR||||"vmsc\\t$dst, $acc1, $s1, $s2,$acc"
VMSC_vmul_cm_core_Y_QX|II_VMSC_vmul_cm_core_Y_QX|0||dst:eDM|acc1:eDM,s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of|||"vmsc\\t$dst, $acc1, $s1, $s2,$acc"
VMSC_vmul_cm_core_Y_QY|II_VMSC_vmul_cm_core_Y_QY|0||dst:eDM|acc1:eDM,s1:eY,s2:eQYs,acc:eR|srSparse_of|||"vmsc\\t$dst, $acc1, $s1, $s2,$acc"
VMSC_vmul_cm_core_Y_X|II_VMSC_vmul_cm_core_Y_X|0||dst:eDM|acc1:eDM,s1:eY,s2:OP_mXw,acc:eR||||"vmsc\\t$dst, $acc1, $s1, $s2,$acc"
VMSC_vmul_cm_core_Y_Y|II_VMSC_vmul_cm_core_Y_Y|0||dst:eDM|acc1:eDM,s1:eY,s2:eY,acc:eR||||"vmsc\\t$dst, $acc1, $s1, $s2,$acc"
VMUL_f_vmul_bf_vmul_bf_core_X_X|II_VMUL_f_vmul_bf_vmul_bf_core_X_X|0||dst:eDM|s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask||"vmul.f\\t$dst, $s1, $s2, $acc"
VMUL_f_vmul_bf_vmul_bf_core_Y_Y|II_VMUL_f_vmul_bf_vmul_bf_core_Y_Y|0||dst:eDM|s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask||"vmul.f\\t$dst, $s1, $s2, $acc"
VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX|II_VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX|0||dst:eDM|s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask||"vmul.f\\t$dst, $s1, $s2, $acc"
VMUL_f_vmul_bfp_vmul_bfp_core_EX_EY|II_VMUL_f_vmul_bfp_vmul_bfp_core_EX_EY|0||dst:eDM|s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask||"vmul.f\\t$dst, $s1, $s2, $acc"
VMUL_f_vmul_bfp_vmul_bfp_core_EX_QEY|II_VMUL_f_vmul_bfp_vmul_bfp_core_EX_QEY|0||dst:eDM|s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask||"vmul.f\\t$dst, $s1, $s2, $acc"
VMUL_f_vmul_bfp_vmul_bfp_core_EY_QEX|II_VMUL_f_vmul_bfp_vmul_bfp_core_EY_QEX|0||dst:eDM|s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask||"vmul.f\\t$dst, $s1, $s2, $acc"
VMUL_vmul_cm_core_X_QX|II_VMUL_vmul_cm_core_X_QX|0||dst:eDM|s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of|||"vmul\\t$dst, $s1, $s2,$acc"
VMUL_vmul_cm_core_X_QY|II_VMUL_vmul_cm_core_X_QY|0||dst:eDM|s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of|||"vmul\\t$dst, $s1, $s2,$acc"
VMUL_vmul_cm_core_X_X|II_VMUL_vmul_cm_core_X_X|0||dst:eDM|s1:OP_mXv,s2:OP_mXw,acc:eR||||"vmul\\t$dst, $s1, $s2,$acc"
VMUL_vmul_cm_core_X_Y|II_VMUL_vmul_cm_core_X_Y|0||dst:eDM|s1:OP_mXv,s2:eY,acc:eR||||"vmul\\t$dst, $s1, $s2,$acc"
VMUL_vmul_cm_core_Y_QX|II_VMUL_vmul_cm_core_Y_QX|0||dst:eDM|s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of|||"vmul\\t$dst, $s1, $s2,$acc"
VMUL_vmul_cm_core_Y_QY|II_VMUL_vmul_cm_core_Y_QY|0||dst:eDM|s1:eY,s2:eQYs,acc:eR|srSparse_of|||"vmul\\t$dst, $s1, $s2,$acc"
VMUL_vmul_cm_core_Y_X|II_VMUL_vmul_cm_core_Y_X|0||dst:eDM|s1:eY,s2:OP_mXw,acc:eR||||"vmul\\t$dst, $s1, $s2,$acc"
VMUL_vmul_cm_core_Y_Y|II_VMUL_vmul_cm_core_Y_Y|0||dst:eDM|s1:eY,s2:eY,acc:eR||||"vmul\\t$dst, $s1, $s2,$acc"
VNEG|II_VNEG|0||dst:eDM|acc1:eDM,acc:eR||||"vneg\\t$dst, $acc1, $acc"
VNEGMUL_f_vmul_bf_vmul_bf_core_X_X|II_VNEGMUL_f_vmul_bf_vmul_bf_core_X_X|0||dst:eDM|s1:OP_mXv,s2:OP_mXw,acc:eR|srFPFlags|crFPMask||"vnegmul.f\\t$dst, $s1, $s2, $acc"
VNEGMUL_f_vmul_bf_vmul_bf_core_Y_Y|II_VNEGMUL_f_vmul_bf_vmul_bf_core_Y_Y|0||dst:eDM|s1:eY,s2:eY,acc:eR|srFPFlags|crFPMask||"vnegmul.f\\t$dst, $s1, $s2, $acc"
VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_EX|II_VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_EX|0||dst:eDM|s1:OP_mEXv,s2:OP_mEXw,acc:eR|srFPFlags|crFPMask||"vnegmul.f\\t$dst, $s1, $s2, $acc"
VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_EY|II_VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_EY|0||dst:eDM|s1:OP_mEXv,s2:eEY,acc:eR|srFPFlags|crFPMask||"vnegmul.f\\t$dst, $s1, $s2, $acc"
VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_QEY|II_VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_QEY|0||dst:eDM|s1:OP_mEXv,s2:eQEYs,acc:eR|srFPFlags,srSparse_of|crFPMask||"vnegmul.f\\t$dst, $s1, $s2, $acc"
VNEGMUL_f_vmul_bfp_vmul_bfp_core_EY_QEX|II_VNEGMUL_f_vmul_bfp_vmul_bfp_core_EY_QEX|0||dst:eDM|s1:eEY,s2:OP_mQEXsw,acc:eR|srFPFlags,srSparse_of|crFPMask||"vnegmul.f\\t$dst, $s1, $s2, $acc"
VNEGMUL_vmul_cm_core_X_QX|II_VNEGMUL_vmul_cm_core_X_QX|0||dst:eDM|s1:OP_mXv,s2:OP_mQXsw,acc:eR|srSparse_of|||"vnegmul\\t$dst, $s1, $s2,$acc"
VNEGMUL_vmul_cm_core_X_QY|II_VNEGMUL_vmul_cm_core_X_QY|0||dst:eDM|s1:OP_mXv,s2:eQYs,acc:eR|srSparse_of|||"vnegmul\\t$dst, $s1, $s2,$acc"
VNEGMUL_vmul_cm_core_X_X|II_VNEGMUL_vmul_cm_core_X_X|0||dst:eDM|s1:OP_mXv,s2:OP_mXw,acc:eR||||"vnegmul\\t$dst, $s1, $s2,$acc"
VNEGMUL_vmul_cm_core_X_Y|II_VNEGMUL_vmul_cm_core_X_Y|0||dst:eDM|s1:OP_mXv,s2:eY,acc:eR||||"vnegmul\\t$dst, $s1, $s2,$acc"
VNEGMUL_vmul_cm_core_Y_QX|II_VNEGMUL_vmul_cm_core_Y_QX|0||dst:eDM|s1:eY,s2:OP_mQXsw,acc:eR|srSparse_of|||"vnegmul\\t$dst, $s1, $s2,$acc"
VNEGMUL_vmul_cm_core_Y_QY|II_VNEGMUL_vmul_cm_core_Y_QY|0||dst:eDM|s1:eY,s2:eQYs,acc:eR|srSparse_of|||"vnegmul\\t$dst, $s1, $s2,$acc"
VNEGMUL_vmul_cm_core_Y_X|II_VNEGMUL_vmul_cm_core_Y_X|0||dst:eDM|s1:eY,s2:OP_mXw,acc:eR||||"vnegmul\\t$dst, $s1, $s2,$acc"
VNEGMUL_vmul_cm_core_Y_Y|II_VNEGMUL_vmul_cm_core_Y_Y|0||dst:eDM|s1:eY,s2:eY,acc:eR||||"vnegmul\\t$dst, $s1, $s2,$acc"
VNEG_GTZ_16|II_VNEG_GTZ_16|0||d:OP_mXm,cmp:mR16_vcompare|s2:OP_mXn||||"vneg_gtz.16\\t $d, r16, $s2"
VNEG_GTZ_32|II_VNEG_GTZ_32|0||d:OP_mXm,cmp:mR16_vcompare|s2:OP_mXn||||"vneg_gtz.32\\t $d, r16, $s2"
VNEG_GTZ_8|II_VNEG_GTZ_8|0||d:OP_mXm,cmp:mL8m|s2:OP_mXn||||"vneg_gtz.8\\t $d, r17:r16, $s2"
VNEG_f|II_VNEG_f|0||dst:eDM|acc1:eDM,acc:eR|srFPFlags|crFPMask||"vneg.f\\t$dst, $acc1, $acc"
VPACK_mv_pack_w_packSign0|II_VPACK_mv_pack_w_packSign0|0||dst:OP_mWa|src:OP_mXs||crPackSize,crSat,packSign0||"vpack\\t$dst, $src, packsign0"
VPACK_mv_pack_w_packSign1|II_VPACK_mv_pack_w_packSign1|0||dst:OP_mWa|src:OP_mXs||crPackSize,crSat,packSign1||"vpack\\t$dst, $src, packsign1"
VPACK_mv_pack_x_packSign0|II_VPACK_mv_pack_x_packSign0|0||dst:OP_mXa|src:eY||crPackSize,crSat,packSign0||"vpack\\t$dst, $src, packsign0"
VPACK_mv_pack_x_packSign1|II_VPACK_mv_pack_x_packSign1|0||dst:OP_mXa|src:eY||crPackSize,crSat,packSign1||"vpack\\t$dst, $src, packsign1"
VPUSH_hi_16|II_VPUSH_hi_16|0||d:OP_mXm|s1:OP_mXm,s2:eR||||"vpush.hi.16\\t $d, $s1, $s2"
VPUSH_hi_32|II_VPUSH_hi_32|0||d:OP_mXm|s1:OP_mXm,s2:eR||||"vpush.hi.32\\t $d, $s1, $s2"
VPUSH_hi_64|II_VPUSH_hi_64|0||d:OP_mXm|s1:OP_mXm,s2:eL||||"vpush.hi.64\\t $d, $s1, $s2"
VPUSH_hi_8|II_VPUSH_hi_8|0||d:OP_mXm|s1:OP_mXm,s2:eR||||"vpush.hi.8\\t $d, $s1, $s2"
VPUSH_lo_16|II_VPUSH_lo_16|0||d:OP_mXm|s1:eR,s2:OP_mXn||||"vpush.lo.16\\t $d, $s1, $s2"
VPUSH_lo_32|II_VPUSH_lo_32|0||d:OP_mXm|s1:eR,s2:OP_mXn||||"vpush.lo.32\\t $d, $s1, $s2"
VPUSH_lo_64|II_VPUSH_lo_64|0||d:OP_mXm|s1:eL,s2:OP_mXn||||"vpush.lo.64\\t $d, $s1, $s2"
VPUSH_lo_8|II_VPUSH_lo_8|0||d:OP_mXm|s1:eR,s2:OP_mXn||||"vpush.lo.8\\t $d, $s1, $s2"
VSEL_16|II_VSEL_16|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn,sel:eRS16||||"vsel.16\\t $d, $s1, $s2, $sel"
VSEL_32|II_VSEL_32|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn,sel:eRS16||||"vsel.32\\t $d, $s1, $s2, $sel"
VSEL_8|II_VSEL_8|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn,sel:eL||||"vsel.8\\t $d, $s1, $s2, $sel"
VSHIFT|II_VSHIFT|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn,shift:eR||||"vshift\\t$d, $s1, $s2, $shift"
VSHIFT_ALIGN|II_VSHIFT_ALIGN|0||d:OP_mXm|s1:OP_mXm,pre:eS,s2:OP_mXn,shift:mR30_shiftx||||"vshift.align\\t$d, $s1, $pre, $s2, r30"
VSHUFFLE_vec_shuffle_bm|II_VSHUFFLE_vec_shuffle_bm|0||dst:OP_mShflBMDst|s1:OP_mXm,s2:OP_mXn,mod:eR||||"vshuffle\\t$dst, $s1, $s2, $mod"
VSHUFFLE_vec_shuffle_ex|II_VSHUFFLE_vec_shuffle_ex|0||dst:OP_mEXm|s1:OP_mEXm,s2:OP_mEXn,mod:eR||||"vshuffle\\t$dst, $s1, $s2, $mod"
VSHUFFLE_vec_shuffle_x|II_VSHUFFLE_vec_shuffle_x|0||dst:OP_mShflXDst|s1:OP_mXm,s2:OP_mXn,mod:eR||||"vshuffle\\t$dst, $s1, $s2, $mod"
VSRS_2x_mv_w_srs_bm_srsSign0|II_VSRS_2x_mv_w_srs_bm_srsSign0|0||dst:OP_mWa|src:OP_mBMs,su:eS|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vsrs.2x\\t$dst, $src, $su, srssign0"
VSRS_2x_mv_w_srs_bm_srsSign1|II_VSRS_2x_mv_w_srs_bm_srsSign1|0||dst:OP_mWa|src:OP_mBMs,su:eS|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vsrs.2x\\t$dst, $src, $su, srssign1"
VSRS_2x_mv_x_srs_cm_srsSign0|II_VSRS_2x_mv_x_srs_cm_srsSign0|0||dst:OP_mXa|src:OP_mCMs,su:eS|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vsrs.2x\\t$dst, $src, $su, srssign0"
VSRS_2x_mv_x_srs_cm_srsSign1|II_VSRS_2x_mv_x_srs_cm_srsSign1|0||dst:OP_mXa|src:OP_mCMs,su:eS|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vsrs.2x\\t$dst, $src, $su, srssign1"
VSRS_4x_mv_w_srs_cm_srsSign0|II_VSRS_4x_mv_w_srs_cm_srsSign0|0||dst:OP_mWa|src:OP_mCMs,su:eS|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vsrs.4x\\t$dst, $src, $su, srssign0"
VSRS_4x_mv_w_srs_cm_srsSign1|II_VSRS_4x_mv_w_srs_cm_srsSign1|0||dst:OP_mWa|src:OP_mCMs,su:eS|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vsrs.4x\\t$dst, $src, $su, srssign1"
VSRS_4x_mv_x_srs_dm_srsSign0|II_VSRS_4x_mv_x_srs_dm_srsSign0|0||dst:OP_mXa|src:eDM,su:eS|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vsrs.4x\\t$dst, $src, $su, srssign0"
VSRS_4x_mv_x_srs_dm_srsSign1|II_VSRS_4x_mv_x_srs_dm_srsSign1|0||dst:OP_mXa|src:eDM,su:eS|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vsrs.4x\\t$dst, $src, $su, srssign1"
VST_128_dmv_sts_w_idx|II_VST_128_dmv_sts_w_idx|8000|||src:OP_mWs,ptr:eP,dj:eDJ||||"vst.128\\t $src, [$ptr, $dj]"
VST_128_dmv_sts_w_idx_imm|II_VST_128_dmv_sts_w_idx_imm|8000|||src:OP_mWs,ptr:eP,imm:c8s_step16||||"vst.128\\t $src, [$ptr, #$imm]"
VST_128_dmv_sts_w_pstm_nrm|II_VST_128_dmv_sts_w_pstm_nrm|8000||ptr_out:eP|src:OP_mWs,ptr:eP,mod:eM|||ptr_out=ptr|"vst.128\\t $src, [$ptr], $mod"
VST_128_dmv_sts_w_pstm_nrm_imm|II_VST_128_dmv_sts_w_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mWs,ptr:eP,imm:c8s_step16|||ptr_out=ptr|"vst.128\\t $src, [$ptr], #$imm"
VST_128_dmv_sts_w_spill|II_VST_128_dmv_sts_w_spill|8000|||src:OP_mWs,imm:c14n_step16||sp||"vst.128\\t $src, [sp, #$imm]"
VST_2D_128|II_VST_2D_128|8006||ptr_out:eP,dc:eDC|src:OP_mWs,ptr:eP,mod:eD|||ptr_out=ptr|"vst.2d.128\\t $src, [$ptr], $mod"
VST_2D_CONV_bf16_fp32_dmw_sts_srs_bf|II_VST_2D_CONV_bf16_fp32_dmw_sts_srs_bf|8006||ptr_out:eP,dc:eDC|src:OP_mBMs,ptr:eP,mod:eD|srF2FFlags|crF2FMask,crRnd|ptr_out=ptr|"vst.2d.conv.bf16.fp32\\t $src, [$ptr], $mod"
VST_2D_CONV_bf16_fp32_dmx_sts_srs_bf|II_VST_2D_CONV_bf16_fp32_dmx_sts_srs_bf|8006||ptr_out:eP,dc:eDC|src:OP_mCMs,ptr:eP,mod:eD|srF2FFlags|crF2FMask,crRnd|ptr_out=ptr|"vst.2d.conv.bf16.fp32\\t $src, [$ptr], $mod"
VST_2D_PACK_dmw_sts_pack_packSign0|II_VST_2D_PACK_dmw_sts_pack_packSign0|8006||ptr_out:eP,dc:eDC|src:OP_mXs,ptr:eP,mod:eD||crPackSize,crSat,packSign0|ptr_out=ptr|"vst.2d.pack\\t$src, packsign0, [$ptr], $mod"
VST_2D_PACK_dmw_sts_pack_packSign1|II_VST_2D_PACK_dmw_sts_pack_packSign1|8006||ptr_out:eP,dc:eDC|src:OP_mXs,ptr:eP,mod:eD||crPackSize,crSat,packSign1|ptr_out=ptr|"vst.2d.pack\\t$src, packsign1, [$ptr], $mod"
VST_2D_PACK_dmx_sts_pack_packSign0|II_VST_2D_PACK_dmx_sts_pack_packSign0|8006||ptr_out:eP,dc:eDC|src:eY,ptr:eP,mod:eD||crPackSize,crSat,packSign0|ptr_out=ptr|"vst.2d.pack\\t$src, packsign0, [$ptr], $mod"
VST_2D_PACK_dmx_sts_pack_packSign1|II_VST_2D_PACK_dmx_sts_pack_packSign1|8006||ptr_out:eP,dc:eDC|src:eY,ptr:eP,mod:eD||crPackSize,crSat,packSign1|ptr_out=ptr|"vst.2d.pack\\t$src, packsign1, [$ptr], $mod"
VST_2D_SRS_2x_dm_sts_srs_cm_srsSign0|II_VST_2D_SRS_2x_dm_sts_srs_cm_srsSign0|8006||ptr_out:eP,dc:eDC|src:OP_mCMs,su:eS,ptr:eP,mod:eD|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.2d.srs.2x\\t $src, $su, srssign0, [$ptr], $mod"
VST_2D_SRS_2x_dm_sts_srs_cm_srsSign1|II_VST_2D_SRS_2x_dm_sts_srs_cm_srsSign1|8006||ptr_out:eP,dc:eDC|src:OP_mCMs,su:eS,ptr:eP,mod:eD|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.2d.srs.2x\\t $src, $su, srssign1, [$ptr], $mod"
VST_2D_SRS_2x_dmw_sts_srs_bm_srsSign0|II_VST_2D_SRS_2x_dmw_sts_srs_bm_srsSign0|8006||ptr_out:eP,dc:eDC|src:OP_mBMs,su:eS,ptr:eP,mod:eD|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.2d.srs.2x\\t $src, $su, srssign0, [$ptr], $mod"
VST_2D_SRS_2x_dmw_sts_srs_bm_srsSign1|II_VST_2D_SRS_2x_dmw_sts_srs_bm_srsSign1|8006||ptr_out:eP,dc:eDC|src:OP_mBMs,su:eS,ptr:eP,mod:eD|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.2d.srs.2x\\t $src, $su, srssign1, [$ptr], $mod"
VST_2D_SRS_4x_dm_sts_srs_cm_srsSign0|II_VST_2D_SRS_4x_dm_sts_srs_cm_srsSign0|8006||ptr_out:eP,dc:eDC|src:OP_mCMs,su:eS,ptr:eP,mod:eD|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.2d.srs.4x\\t $src, $su, srssign0, [$ptr], $mod"
VST_2D_SRS_4x_dm_sts_srs_cm_srsSign1|II_VST_2D_SRS_4x_dm_sts_srs_cm_srsSign1|8006||ptr_out:eP,dc:eDC|src:OP_mCMs,su:eS,ptr:eP,mod:eD|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.2d.srs.4x\\t $src, $su, srssign1, [$ptr], $mod"
VST_2D_SRS_4x_dmx_sts_srs_dm_srsSign0|II_VST_2D_SRS_4x_dmx_sts_srs_dm_srsSign0|8006||ptr_out:eP,dc:eDC|src:eDM,su:eS,ptr:eP,mod:eD|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.2d.srs.4x\\t $src, $su, srssign0, [$ptr], $mod"
VST_2D_SRS_4x_dmx_sts_srs_dm_srsSign1|II_VST_2D_SRS_4x_dmx_sts_srs_dm_srsSign1|8006||ptr_out:eP,dc:eDC|src:eDM,su:eS,ptr:eP,mod:eD|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.2d.srs.4x\\t $src, $su, srssign1, [$ptr], $mod"
VST_2D_dmw_sts_w|II_VST_2D_dmw_sts_w|8006||ptr_out:eP,dc:eDC|src:OP_mWs,ptr:eP,mod:eD|||ptr_out=ptr|"vst.2d\\t $src, [$ptr], $mod"
VST_2D_dmx_sts_bm|II_VST_2D_dmx_sts_bm|8006||ptr_out:eP,dc:eDC|src:OP_mBMs,ptr:eP,mod:eD|||ptr_out=ptr|"vst.2d\\t $src, [$ptr], $mod"
VST_2D_dmx_sts_fifohl|II_VST_2D_dmx_sts_fifohl|8006||ptr_out:eP,dc:eDC|src:OP_mFifoHLReg,ptr:eP,mod:eD|||ptr_out=ptr|"vst.2d\\t $src, [$ptr], $mod"
VST_2D_dmx_sts_x|II_VST_2D_dmx_sts_x|8006||ptr_out:eP,dc:eDC|src:OP_mXs,ptr:eP,mod:eD|||ptr_out=ptr|"vst.2d\\t $src, [$ptr], $mod"
VST_3D_128|II_VST_3D_128|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mWs,ptr:eP,mod:eDS|||ptr_out=ptr|"vst.3d.128\\t $src, [$ptr], $mod"
VST_3D_CONV_bf16_fp32_dmw_sts_srs_bf|II_VST_3D_CONV_bf16_fp32_dmw_sts_srs_bf|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mBMs,ptr:eP,mod:eDS|srF2FFlags|crF2FMask,crRnd|ptr_out=ptr|"vst.3d.conv.bf16.fp32\\t $src, [$ptr], $mod"
VST_3D_CONV_bf16_fp32_dmx_sts_srs_bf|II_VST_3D_CONV_bf16_fp32_dmx_sts_srs_bf|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mCMs,ptr:eP,mod:eDS|srF2FFlags|crF2FMask,crRnd|ptr_out=ptr|"vst.3d.conv.bf16.fp32\\t $src, [$ptr], $mod"
VST_3D_PACK_dmw_sts_pack_packSign0|II_VST_3D_PACK_dmw_sts_pack_packSign0|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mXs,ptr:eP,mod:eDS||crPackSize,crSat,packSign0|ptr_out=ptr|"vst.3d.pack\\t$src, packsign0, [$ptr], $mod"
VST_3D_PACK_dmw_sts_pack_packSign1|II_VST_3D_PACK_dmw_sts_pack_packSign1|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mXs,ptr:eP,mod:eDS||crPackSize,crSat,packSign1|ptr_out=ptr|"vst.3d.pack\\t$src, packsign1, [$ptr], $mod"
VST_3D_PACK_dmx_sts_pack_packSign0|II_VST_3D_PACK_dmx_sts_pack_packSign0|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:eY,ptr:eP,mod:eDS||crPackSize,crSat,packSign0|ptr_out=ptr|"vst.3d.pack\\t$src, packsign0, [$ptr], $mod"
VST_3D_PACK_dmx_sts_pack_packSign1|II_VST_3D_PACK_dmx_sts_pack_packSign1|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:eY,ptr:eP,mod:eDS||crPackSize,crSat,packSign1|ptr_out=ptr|"vst.3d.pack\\t$src, packsign1, [$ptr], $mod"
VST_3D_SRS_2x_dm_sts_srs_cm_srsSign0|II_VST_3D_SRS_2x_dm_sts_srs_cm_srsSign0|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mCMs,su:eS,ptr:eP,mod:eDS|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.3d.srs.2x\\t $src, $su, srssign0, [$ptr], $mod"
VST_3D_SRS_2x_dm_sts_srs_cm_srsSign1|II_VST_3D_SRS_2x_dm_sts_srs_cm_srsSign1|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mCMs,su:eS,ptr:eP,mod:eDS|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.3d.srs.2x\\t $src, $su, srssign1, [$ptr], $mod"
VST_3D_SRS_2x_dmw_sts_srs_bm_srsSign0|II_VST_3D_SRS_2x_dmw_sts_srs_bm_srsSign0|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mBMs,su:eS,ptr:eP,mod:eDS|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.3d.srs.2x\\t $src, $su, srssign0, [$ptr], $mod"
VST_3D_SRS_2x_dmw_sts_srs_bm_srsSign1|II_VST_3D_SRS_2x_dmw_sts_srs_bm_srsSign1|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mBMs,su:eS,ptr:eP,mod:eDS|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.3d.srs.2x\\t $src, $su, srssign1, [$ptr], $mod"
VST_3D_SRS_4x_dm_sts_srs_cm_srsSign0|II_VST_3D_SRS_4x_dm_sts_srs_cm_srsSign0|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mCMs,su:eS,ptr:eP,mod:eDS|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.3d.srs.4x\\t $src, $su, srssign0, [$ptr], $mod"
VST_3D_SRS_4x_dm_sts_srs_cm_srsSign1|II_VST_3D_SRS_4x_dm_sts_srs_cm_srsSign1|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mCMs,su:eS,ptr:eP,mod:eDS|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.3d.srs.4x\\t $src, $su, srssign1, [$ptr], $mod"
VST_3D_SRS_4x_dmx_sts_srs_dm_srsSign0|II_VST_3D_SRS_4x_dmx_sts_srs_dm_srsSign0|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:eDM,su:eS,ptr:eP,mod:eDS|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.3d.srs.4x\\t $src, $su, srssign0, [$ptr], $mod"
VST_3D_SRS_4x_dmx_sts_srs_dm_srsSign1|II_VST_3D_SRS_4x_dmx_sts_srs_dm_srsSign1|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:eDM,su:eS,ptr:eP,mod:eDS|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.3d.srs.4x\\t $src, $su, srssign1, [$ptr], $mod"
VST_3D_dmw_sts_w|II_VST_3D_dmw_sts_w|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mWs,ptr:eP,mod:eDS|||ptr_out=ptr|"vst.3d\\t $src, [$ptr], $mod"
VST_3D_dmx_sts_bm|II_VST_3D_dmx_sts_bm|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mBMs,ptr:eP,mod:eDS|||ptr_out=ptr|"vst.3d\\t $src, [$ptr], $mod"
VST_3D_dmx_sts_fifohl|II_VST_3D_dmx_sts_fifohl|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mFifoHLReg,ptr:eP,mod:eDS|||ptr_out=ptr|"vst.3d\\t $src, [$ptr], $mod"
VST_3D_dmx_sts_x|II_VST_3D_dmx_sts_x|8006||ptr_out:eP,dcl:eDCL,dch:eDCH|src:OP_mXs,ptr:eP,mod:eDS|||ptr_out=ptr|"vst.3d\\t $src, [$ptr], $mod"
VST_CONV_bf16_fp32_dmw_sts_srs_bf_idx|II_VST_CONV_bf16_fp32_dmw_sts_srs_bf_idx|8000|||src:OP_mBMs,ptr:eP,dj:eDJ|srF2FFlags|crF2FMask,crRnd||"vst.conv.bf16.fp32\\t $src, [$ptr, $dj]"
VST_CONV_bf16_fp32_dmw_sts_srs_bf_idx_imm|II_VST_CONV_bf16_fp32_dmw_sts_srs_bf_idx_imm|8000|||src:OP_mBMs,ptr:eP,imm:c9s_step32|srF2FFlags|crF2FMask,crRnd||"vst.conv.bf16.fp32\\t $src, [$ptr, #$imm]"
VST_CONV_bf16_fp32_dmw_sts_srs_bf_pstm_nrm|II_VST_CONV_bf16_fp32_dmw_sts_srs_bf_pstm_nrm|8000||ptr_out:eP|src:OP_mBMs,ptr:eP,mod:eM|srF2FFlags|crF2FMask,crRnd|ptr_out=ptr|"vst.conv.bf16.fp32\\t $src, [$ptr], $mod"
VST_CONV_bf16_fp32_dmw_sts_srs_bf_pstm_nrm_imm|II_VST_CONV_bf16_fp32_dmw_sts_srs_bf_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mBMs,ptr:eP,imm:c9s_step32|srF2FFlags|crF2FMask,crRnd|ptr_out=ptr|"vst.conv.bf16.fp32\\t $src, [$ptr], #$imm"
VST_CONV_bf16_fp32_dmx_sts_srs_bf_idx|II_VST_CONV_bf16_fp32_dmx_sts_srs_bf_idx|8000|||src:OP_mCMs,ptr:eP,dj:eDJ|srF2FFlags|crF2FMask,crRnd||"vst.conv.bf16.fp32\\t $src, [$ptr, $dj]"
VST_CONV_bf16_fp32_dmx_sts_srs_bf_idx_imm|II_VST_CONV_bf16_fp32_dmx_sts_srs_bf_idx_imm|8000|||src:OP_mCMs,ptr:eP,imm:c10s_step64|srF2FFlags|crF2FMask,crRnd||"vst.conv.bf16.fp32\\t $src, [$ptr, #$imm]"
VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm|II_VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm|8000||ptr_out:eP|src:OP_mCMs,ptr:eP,mod:eM|srF2FFlags|crF2FMask,crRnd|ptr_out=ptr|"vst.conv.bf16.fp32\\t $src, [$ptr], $mod"
VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm|II_VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mCMs,ptr:eP,imm:c10s_step64|srF2FFlags|crF2FMask,crRnd|ptr_out=ptr|"vst.conv.bf16.fp32\\t $src, [$ptr], #$imm"
VST_FLUSH_512_2D|II_VST_FLUSH_512_2D|8006||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st,dc:eDC|fifo_reg:mStFifo,ptr:mPfs,avail:mR26_fifo_st,mod:eD|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.flush.512.2d\\t [p2, sf, r26, $mod]"
VST_FLUSH_512_3D|II_VST_FLUSH_512_3D|8006||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st,dcl:eDCL,dch:eDCH|fifo_reg:mStFifo,ptr:mPfs,avail:mR26_fifo_st,mod:eDS|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.flush.512.3d\\t [p2, sf, r26, $mod]"
VST_FLUSH_512_CONV_2D|II_VST_FLUSH_512_CONV_2D|8006||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st,dc:eDC|fifo_reg:mStFifo,ptr:mPfs,avail:mR26_fifo_st,mod:eD|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.flush.512.conv.2d\\t [p2, sf, r26, $mod]"
VST_FLUSH_512_CONV_3D|II_VST_FLUSH_512_CONV_3D|8006||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st,dcl:eDCL,dch:eDCH|fifo_reg:mStFifo,ptr:mPfs,avail:mR26_fifo_st,mod:eDS|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.flush.512.conv.3d\\t [p2, sf, r26, $mod]"
VST_FLUSH_512_CONV_fifo_1d_flush|II_VST_FLUSH_512_CONV_fifo_1d_flush|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,ptr:mPfs,avail:mR26_fifo_st,mod:eM|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.flush.512.conv\\t [p2, sf, r26, $mod]"
VST_FLUSH_512_CONV_normal_flush|II_VST_FLUSH_512_CONV_normal_flush|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,ptr:mPfs,avail:mR26_fifo_st|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.flush.512.conv\\t [p2, sf, r26]"
VST_FLUSH_512_fifo_1d_flush|II_VST_FLUSH_512_fifo_1d_flush|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,ptr:mPfs,avail:mR26_fifo_st,mod:eM|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.flush.512\\t [p2, sf, r26, $mod]"
VST_FLUSH_512_normal_flush|II_VST_FLUSH_512_normal_flush|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,ptr:mPfs,avail:mR26_fifo_st|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.flush.512\\t [p2, sf, r26]"
VST_PACK_dmw_sts_pack_idx_imm_packSign0|II_VST_PACK_dmw_sts_pack_idx_imm_packSign0|8000|||src:OP_mXs,ptr:eP,imm:c9s_step32||crPackSize,crSat,packSign0||"vst.pack\\t$src, packsign0, [$ptr, #$imm]"
VST_PACK_dmw_sts_pack_idx_imm_packSign1|II_VST_PACK_dmw_sts_pack_idx_imm_packSign1|8000|||src:OP_mXs,ptr:eP,imm:c9s_step32||crPackSize,crSat,packSign1||"vst.pack\\t$src, packsign1, [$ptr, #$imm]"
VST_PACK_dmw_sts_pack_idx_packSign0|II_VST_PACK_dmw_sts_pack_idx_packSign0|8000|||src:OP_mXs,ptr:eP,dj:eDJ||crPackSize,crSat,packSign0||"vst.pack\\t$src, packsign0, [$ptr, $dj]"
VST_PACK_dmw_sts_pack_idx_packSign1|II_VST_PACK_dmw_sts_pack_idx_packSign1|8000|||src:OP_mXs,ptr:eP,dj:eDJ||crPackSize,crSat,packSign1||"vst.pack\\t$src, packsign1, [$ptr, $dj]"
VST_PACK_dmw_sts_pack_pstm_nrm_imm_packSign0|II_VST_PACK_dmw_sts_pack_pstm_nrm_imm_packSign0|8000||ptr_out:eP|src:OP_mXs,ptr:eP,imm:c9s_step32||crPackSize,crSat,packSign0|ptr_out=ptr|"vst.pack\\t$src, packsign0, [$ptr], #$imm"
VST_PACK_dmw_sts_pack_pstm_nrm_imm_packSign1|II_VST_PACK_dmw_sts_pack_pstm_nrm_imm_packSign1|8000||ptr_out:eP|src:OP_mXs,ptr:eP,imm:c9s_step32||crPackSize,crSat,packSign1|ptr_out=ptr|"vst.pack\\t$src, packsign1, [$ptr], #$imm"
VST_PACK_dmw_sts_pack_pstm_nrm_packSign0|II_VST_PACK_dmw_sts_pack_pstm_nrm_packSign0|8000||ptr_out:eP|src:OP_mXs,ptr:eP,mod:eM||crPackSize,crSat,packSign0|ptr_out=ptr|"vst.pack\\t$src, packsign0, [$ptr], $mod"
VST_PACK_dmw_sts_pack_pstm_nrm_packSign1|II_VST_PACK_dmw_sts_pack_pstm_nrm_packSign1|8000||ptr_out:eP|src:OP_mXs,ptr:eP,mod:eM||crPackSize,crSat,packSign1|ptr_out=ptr|"vst.pack\\t$src, packsign1, [$ptr], $mod"
VST_PACK_dmx_sts_pack_idx_imm_packSign0|II_VST_PACK_dmx_sts_pack_idx_imm_packSign0|8000|||src:eY,ptr:eP,imm:c10s_step64||crPackSize,crSat,packSign0||"vst.pack\\t$src, packsign0, [$ptr, #$imm]"
VST_PACK_dmx_sts_pack_idx_imm_packSign1|II_VST_PACK_dmx_sts_pack_idx_imm_packSign1|8000|||src:eY,ptr:eP,imm:c10s_step64||crPackSize,crSat,packSign1||"vst.pack\\t$src, packsign1, [$ptr, #$imm]"
VST_PACK_dmx_sts_pack_idx_packSign0|II_VST_PACK_dmx_sts_pack_idx_packSign0|8000|||src:eY,ptr:eP,dj:eDJ||crPackSize,crSat,packSign0||"vst.pack\\t$src, packsign0, [$ptr, $dj]"
VST_PACK_dmx_sts_pack_idx_packSign1|II_VST_PACK_dmx_sts_pack_idx_packSign1|8000|||src:eY,ptr:eP,dj:eDJ||crPackSize,crSat,packSign1||"vst.pack\\t$src, packsign1, [$ptr, $dj]"
VST_PACK_dmx_sts_pack_pstm_nrm_imm_packSign0|II_VST_PACK_dmx_sts_pack_pstm_nrm_imm_packSign0|8000||ptr_out:eP|src:eY,ptr:eP,imm:c10s_step64||crPackSize,crSat,packSign0|ptr_out=ptr|"vst.pack\\t$src, packsign0, [$ptr], #$imm"
VST_PACK_dmx_sts_pack_pstm_nrm_imm_packSign1|II_VST_PACK_dmx_sts_pack_pstm_nrm_imm_packSign1|8000||ptr_out:eP|src:eY,ptr:eP,imm:c10s_step64||crPackSize,crSat,packSign1|ptr_out=ptr|"vst.pack\\t$src, packsign1, [$ptr], #$imm"
VST_PACK_dmx_sts_pack_pstm_nrm_packSign0|II_VST_PACK_dmx_sts_pack_pstm_nrm_packSign0|8000||ptr_out:eP|src:eY,ptr:eP,mod:eM||crPackSize,crSat,packSign0|ptr_out=ptr|"vst.pack\\t$src, packsign0, [$ptr], $mod"
VST_PACK_dmx_sts_pack_pstm_nrm_packSign1|II_VST_PACK_dmx_sts_pack_pstm_nrm_packSign1|8000||ptr_out:eP|src:eY,ptr:eP,mod:eM||crPackSize,crSat,packSign1|ptr_out=ptr|"vst.pack\\t$src, packsign1, [$ptr], $mod"
VST_PUSH_512|II_VST_PUSH_512|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,src:OP_mXs,ptr:mPfs,avail:mR26_fifo_st|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.push.512\\t $src, [p2, sf, r26]"
VST_PUSH_544|II_VST_PUSH_544|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,src:OP_mEXs,ptr:mPfs,avail:mR26_fifo_st|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.push.544\\t $src, [p2, sf, r26]"
VST_PUSH_544_CONV_bfp16ebs16_ebs8|II_VST_PUSH_544_CONV_bfp16ebs16_ebs8|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,src:OP_mEXs,ptr:mPfs,avail:mR26_fifo_st|srF2BFlags,srFifo_of|crF2BMask,crRnd|fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.push.544.conv.bfp16ebs16.ebs8\\t $src, [p2, sf, r26]"
VST_PUSH_544_CONV_bfp16ebs16_fp32|II_VST_PUSH_544_CONV_bfp16ebs16_fp32|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,src:eDM,ptr:mPfs,avail:mR26_fifo_st|srF2BFlags,srFifo_of|crF2BMask,crRnd|fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.push.544.conv.bfp16ebs16.fp32\\t $src, [p2, sf, r26]"
VST_PUSH_576|II_VST_PUSH_576|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,src:OP_mEXs,ptr:mPfs,avail:mR26_fifo_st|srFifo_of||fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.push.576\\t $src, [p2, sf, r26]"
VST_PUSH_576_CONV_bfp16ebs8_fp32|II_VST_PUSH_576_CONV_bfp16ebs8_fp32|8000||fifo_reg_out:mStFifo,ptr_out:mPfs,avail_out:mR26_fifo_st|fifo_reg:mStFifo,src:eDM,ptr:mPfs,avail:mR26_fifo_st|srF2BFlags,srFifo_of|crF2BMask,crRnd|fifo_reg_out=fifo_reg,ptr_out=ptr,avail_out=avail|"vst.push.576.conv.bfp16ebs8.fp32\\t $src, [p2, sf, r26]"
VST_SRS_2x_dm_sts_srs_cm_idx_imm_srsSign0|II_VST_SRS_2x_dm_sts_srs_cm_idx_imm_srsSign0|8000|||src:OP_mCMs,su:eS,ptr:eP,imm:c10s_step64|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vst.srs.2x\\t $src, $su, srssign0, [$ptr, #$imm]"
VST_SRS_2x_dm_sts_srs_cm_idx_imm_srsSign1|II_VST_SRS_2x_dm_sts_srs_cm_idx_imm_srsSign1|8000|||src:OP_mCMs,su:eS,ptr:eP,imm:c10s_step64|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vst.srs.2x\\t $src, $su, srssign1, [$ptr, #$imm]"
VST_SRS_2x_dm_sts_srs_cm_idx_srsSign0|II_VST_SRS_2x_dm_sts_srs_cm_idx_srsSign0|8000|||src:OP_mCMs,su:eS,ptr:eP,dj:eDJ|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vst.srs.2x\\t $src, $su, srssign0, [$ptr, $dj]"
VST_SRS_2x_dm_sts_srs_cm_idx_srsSign1|II_VST_SRS_2x_dm_sts_srs_cm_idx_srsSign1|8000|||src:OP_mCMs,su:eS,ptr:eP,dj:eDJ|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vst.srs.2x\\t $src, $su, srssign1, [$ptr, $dj]"
VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign0|II_VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign0|8000||ptr_out:eP|src:OP_mCMs,su:eS,ptr:eP,imm:c10s_step64|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.srs.2x\\t $src, $su, srssign0, [$ptr], #$imm"
VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1|II_VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1|8000||ptr_out:eP|src:OP_mCMs,su:eS,ptr:eP,imm:c10s_step64|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.srs.2x\\t $src, $su, srssign1, [$ptr], #$imm"
VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_srsSign0|II_VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_srsSign0|8000||ptr_out:eP|src:OP_mCMs,su:eS,ptr:eP,mod:eM|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.srs.2x\\t $src, $su, srssign0, [$ptr], $mod"
VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_srsSign1|II_VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_srsSign1|8000||ptr_out:eP|src:OP_mCMs,su:eS,ptr:eP,mod:eM|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.srs.2x\\t $src, $su, srssign1, [$ptr], $mod"
VST_SRS_2x_dmw_sts_srs_bm_idx_imm_srsSign0|II_VST_SRS_2x_dmw_sts_srs_bm_idx_imm_srsSign0|8000|||src:OP_mBMs,su:eS,ptr:eP,imm:c9s_step32|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vst.srs.2x\\t $src, $su, srssign0, [$ptr, #$imm]"
VST_SRS_2x_dmw_sts_srs_bm_idx_imm_srsSign1|II_VST_SRS_2x_dmw_sts_srs_bm_idx_imm_srsSign1|8000|||src:OP_mBMs,su:eS,ptr:eP,imm:c9s_step32|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vst.srs.2x\\t $src, $su, srssign1, [$ptr, #$imm]"
VST_SRS_2x_dmw_sts_srs_bm_idx_srsSign0|II_VST_SRS_2x_dmw_sts_srs_bm_idx_srsSign0|8000|||src:OP_mBMs,su:eS,ptr:eP,dj:eDJ|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vst.srs.2x\\t $src, $su, srssign0, [$ptr, $dj]"
VST_SRS_2x_dmw_sts_srs_bm_idx_srsSign1|II_VST_SRS_2x_dmw_sts_srs_bm_idx_srsSign1|8000|||src:OP_mBMs,su:eS,ptr:eP,dj:eDJ|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vst.srs.2x\\t $src, $su, srssign1, [$ptr, $dj]"
VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_imm_srsSign0|II_VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_imm_srsSign0|8000||ptr_out:eP|src:OP_mBMs,su:eS,ptr:eP,imm:c9s_step32|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.srs.2x\\t $src, $su, srssign0, [$ptr], #$imm"
VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_imm_srsSign1|II_VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_imm_srsSign1|8000||ptr_out:eP|src:OP_mBMs,su:eS,ptr:eP,imm:c9s_step32|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.srs.2x\\t $src, $su, srssign1, [$ptr], #$imm"
VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_srsSign0|II_VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_srsSign0|8000||ptr_out:eP|src:OP_mBMs,su:eS,ptr:eP,mod:eM|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.srs.2x\\t $src, $su, srssign0, [$ptr], $mod"
VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_srsSign1|II_VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_srsSign1|8000||ptr_out:eP|src:OP_mBMs,su:eS,ptr:eP,mod:eM|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.srs.2x\\t $src, $su, srssign1, [$ptr], $mod"
VST_SRS_4x_dm_sts_srs_cm_idx_imm_srsSign0|II_VST_SRS_4x_dm_sts_srs_cm_idx_imm_srsSign0|8000|||src:OP_mCMs,su:eS,ptr:eP,imm:c9s_step32|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vst.srs.4x\\t $src, $su, srssign0, [$ptr, #$imm]"
VST_SRS_4x_dm_sts_srs_cm_idx_imm_srsSign1|II_VST_SRS_4x_dm_sts_srs_cm_idx_imm_srsSign1|8000|||src:OP_mCMs,su:eS,ptr:eP,imm:c9s_step32|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vst.srs.4x\\t $src, $su, srssign1, [$ptr, #$imm]"
VST_SRS_4x_dm_sts_srs_cm_idx_srsSign0|II_VST_SRS_4x_dm_sts_srs_cm_idx_srsSign0|8000|||src:OP_mCMs,su:eS,ptr:eP,dj:eDJ|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vst.srs.4x\\t $src, $su, srssign0, [$ptr, $dj]"
VST_SRS_4x_dm_sts_srs_cm_idx_srsSign1|II_VST_SRS_4x_dm_sts_srs_cm_idx_srsSign1|8000|||src:OP_mCMs,su:eS,ptr:eP,dj:eDJ|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vst.srs.4x\\t $src, $su, srssign1, [$ptr, $dj]"
VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_imm_srsSign0|II_VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_imm_srsSign0|8000||ptr_out:eP|src:OP_mCMs,su:eS,ptr:eP,imm:c9s_step32|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.srs.4x\\t $src, $su, srssign0, [$ptr], #$imm"
VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1|II_VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1|8000||ptr_out:eP|src:OP_mCMs,su:eS,ptr:eP,imm:c9s_step32|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.srs.4x\\t $src, $su, srssign1, [$ptr], #$imm"
VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_srsSign0|II_VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_srsSign0|8000||ptr_out:eP|src:OP_mCMs,su:eS,ptr:eP,mod:eM|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.srs.4x\\t $src, $su, srssign0, [$ptr], $mod"
VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_srsSign1|II_VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_srsSign1|8000||ptr_out:eP|src:OP_mCMs,su:eS,ptr:eP,mod:eM|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.srs.4x\\t $src, $su, srssign1, [$ptr], $mod"
VST_SRS_4x_dmx_sts_srs_dm_idx_imm_srsSign0|II_VST_SRS_4x_dmx_sts_srs_dm_idx_imm_srsSign0|8000|||src:eDM,su:eS,ptr:eP,imm:c10s_step64|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vst.srs.4x\\t $src, $su, srssign0, [$ptr, #$imm]"
VST_SRS_4x_dmx_sts_srs_dm_idx_imm_srsSign1|II_VST_SRS_4x_dmx_sts_srs_dm_idx_imm_srsSign1|8000|||src:eDM,su:eS,ptr:eP,imm:c10s_step64|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vst.srs.4x\\t $src, $su, srssign1, [$ptr, #$imm]"
VST_SRS_4x_dmx_sts_srs_dm_idx_srsSign0|II_VST_SRS_4x_dmx_sts_srs_dm_idx_srsSign0|8000|||src:eDM,su:eS,ptr:eP,dj:eDJ|srSRS_of|crRnd,crSRSMode,crSat,srsSign0||"vst.srs.4x\\t $src, $su, srssign0, [$ptr, $dj]"
VST_SRS_4x_dmx_sts_srs_dm_idx_srsSign1|II_VST_SRS_4x_dmx_sts_srs_dm_idx_srsSign1|8000|||src:eDM,su:eS,ptr:eP,dj:eDJ|srSRS_of|crRnd,crSRSMode,crSat,srsSign1||"vst.srs.4x\\t $src, $su, srssign1, [$ptr, $dj]"
VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign0|II_VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign0|8000||ptr_out:eP|src:eDM,su:eS,ptr:eP,imm:c10s_step64|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.srs.4x\\t $src, $su, srssign0, [$ptr], #$imm"
VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1|II_VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1|8000||ptr_out:eP|src:eDM,su:eS,ptr:eP,imm:c10s_step64|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.srs.4x\\t $src, $su, srssign1, [$ptr], #$imm"
VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_srsSign0|II_VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_srsSign0|8000||ptr_out:eP|src:eDM,su:eS,ptr:eP,mod:eM|srSRS_of|crRnd,crSRSMode,crSat,srsSign0|ptr_out=ptr|"vst.srs.4x\\t $src, $su, srssign0, [$ptr], $mod"
VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_srsSign1|II_VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_srsSign1|8000||ptr_out:eP|src:eDM,su:eS,ptr:eP,mod:eM|srSRS_of|crRnd,crSRSMode,crSat,srsSign1|ptr_out=ptr|"vst.srs.4x\\t $src, $su, srssign1, [$ptr], $mod"
VST_dmw_sts_w_idx|II_VST_dmw_sts_w_idx|8000|||src:OP_mWs,ptr:eP,dj:eDJ||||"vst\\t $src, [$ptr, $dj]"
VST_dmw_sts_w_idx_imm|II_VST_dmw_sts_w_idx_imm|8000|||src:OP_mWs,ptr:eP,imm:c9s_step32||||"vst\\t $src, [$ptr, #$imm]"
VST_dmw_sts_w_pstm_nrm|II_VST_dmw_sts_w_pstm_nrm|8000||ptr_out:eP|src:OP_mWs,ptr:eP,mod:eM|||ptr_out=ptr|"vst\\t $src, [$ptr], $mod"
VST_dmw_sts_w_pstm_nrm_imm|II_VST_dmw_sts_w_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mWs,ptr:eP,imm:c9s_step32|||ptr_out=ptr|"vst\\t $src, [$ptr], #$imm"
VST_dmw_sts_w_spill|II_VST_dmw_sts_w_spill|8000|||src:OP_mWs,imm:c15n_step32||sp||"vst\\t $src, [sp, #$imm]"
VST_dmx_sts_bm_idx|II_VST_dmx_sts_bm_idx|8000|||src:OP_mBMs,ptr:eP,dj:eDJ||||"vst\\t $src, [$ptr, $dj]"
VST_dmx_sts_bm_idx_imm|II_VST_dmx_sts_bm_idx_imm|8000|||src:OP_mBMs,ptr:eP,imm:c10s_step64||||"vst\\t $src, [$ptr, #$imm]"
VST_dmx_sts_bm_pstm_nrm|II_VST_dmx_sts_bm_pstm_nrm|8000||ptr_out:eP|src:OP_mBMs,ptr:eP,mod:eM|||ptr_out=ptr|"vst\\t $src, [$ptr], $mod"
VST_dmx_sts_bm_pstm_nrm_imm|II_VST_dmx_sts_bm_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mBMs,ptr:eP,imm:c10s_step64|||ptr_out=ptr|"vst\\t $src, [$ptr], #$imm"
VST_dmx_sts_bm_spill|II_VST_dmx_sts_bm_spill|8000|||src:OP_mBMs,imm:c16n_step64||sp||"vst\\t $src, [sp, #$imm]"
VST_dmx_sts_fifohl_idx|II_VST_dmx_sts_fifohl_idx|8000|||src:OP_mFifoHLReg,ptr:eP,dj:eDJ||||"vst\\t $src, [$ptr, $dj]"
VST_dmx_sts_fifohl_idx_imm|II_VST_dmx_sts_fifohl_idx_imm|8000|||src:OP_mFifoHLReg,ptr:eP,imm:c10s_step64||||"vst\\t $src, [$ptr, #$imm]"
VST_dmx_sts_fifohl_pstm_nrm|II_VST_dmx_sts_fifohl_pstm_nrm|8000||ptr_out:eP|src:OP_mFifoHLReg,ptr:eP,mod:eM|||ptr_out=ptr|"vst\\t $src, [$ptr], $mod"
VST_dmx_sts_fifohl_pstm_nrm_imm|II_VST_dmx_sts_fifohl_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mFifoHLReg,ptr:eP,imm:c10s_step64|||ptr_out=ptr|"vst\\t $src, [$ptr], #$imm"
VST_dmx_sts_fifohl_spill|II_VST_dmx_sts_fifohl_spill|8000|||src:OP_mFifoHLReg,imm:c16n_step64||sp||"vst\\t $src, [sp, #$imm]"
VST_dmx_sts_x_idx|II_VST_dmx_sts_x_idx|8000|||src:OP_mXs,ptr:eP,dj:eDJ||||"vst\\t $src, [$ptr, $dj]"
VST_dmx_sts_x_idx_imm|II_VST_dmx_sts_x_idx_imm|8000|||src:OP_mXs,ptr:eP,imm:c10s_step64||||"vst\\t $src, [$ptr, #$imm]"
VST_dmx_sts_x_pstm_nrm|II_VST_dmx_sts_x_pstm_nrm|8000||ptr_out:eP|src:OP_mXs,ptr:eP,mod:eM|||ptr_out=ptr|"vst\\t $src, [$ptr], $mod"
VST_dmx_sts_x_pstm_nrm_imm|II_VST_dmx_sts_x_pstm_nrm_imm|8000||ptr_out:eP|src:OP_mXs,ptr:eP,imm:c10s_step64|||ptr_out=ptr|"vst\\t $src, [$ptr], #$imm"
VST_dmx_sts_x_spill|II_VST_dmx_sts_x_spill|8000|||src:OP_mXs,imm:c16n_step64||sp||"vst\\t $src, [sp, #$imm]"
VSUB_16|II_VSUB_16|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn||||"vsub.16\\t $d, $s1, $s2"
VSUB_32|II_VSUB_32|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn||||"vsub.32\\t $d, $s1, $s2"
VSUB_8|II_VSUB_8|0||d:OP_mXm|s1:OP_mXm,s2:OP_mXn||||"vsub.8\\t $d, $s1, $s2"
VSUB_GE_16_vaddSign0|II_VSUB_GE_16_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vsub_ge.16\\t $d, r16, $s1, $s2, vaddsign0"
VSUB_GE_16_vaddSign1|II_VSUB_GE_16_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vsub_ge.16\\t $d, r16, $s1, $s2, vaddsign1"
VSUB_GE_32_vaddSign0|II_VSUB_GE_32_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vsub_ge.32\\t $d, r16, $s1, $s2, vaddsign0"
VSUB_GE_32_vaddSign1|II_VSUB_GE_32_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vsub_ge.32\\t $d, r16, $s1, $s2, vaddsign1"
VSUB_GE_8_vaddSign0|II_VSUB_GE_8_vaddSign0|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vsub_ge.8\\t $d, r17:r16, $s1, $s2, vaddsign0"
VSUB_GE_8_vaddSign1|II_VSUB_GE_8_vaddSign1|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vsub_ge.8\\t $d, r17:r16, $s1, $s2, vaddsign1"
VSUB_LT_16_vaddSign0|II_VSUB_LT_16_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vsub_lt.16\\t $d, r16, $s1, $s2, vaddsign0"
VSUB_LT_16_vaddSign1|II_VSUB_LT_16_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vsub_lt.16\\t $d, r16, $s1, $s2, vaddsign1"
VSUB_LT_32_vaddSign0|II_VSUB_LT_32_vaddSign0|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vsub_lt.32\\t $d, r16, $s1, $s2, vaddsign0"
VSUB_LT_32_vaddSign1|II_VSUB_LT_32_vaddSign1|0||d:OP_mXm,cmp:mR16_vcompare|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vsub_lt.32\\t $d, r16, $s1, $s2, vaddsign1"
VSUB_LT_8_vaddSign0|II_VSUB_LT_8_vaddSign0|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign0||"vsub_lt.8\\t $d, r17:r16, $s1, $s2, vaddsign0"
VSUB_LT_8_vaddSign1|II_VSUB_LT_8_vaddSign1|0||d:OP_mXm,cmp:mL8m|s1:OP_mXm,s2:OP_mXn||vaddSign1||"vsub_lt.8\\t $d, r17:r16, $s1, $s2, vaddsign1"
VSUB_f_vmac_cm2_add_reg|II_VSUB_f_vmac_cm2_add_reg|0||dst:eDM|acc1:eDM,acc2:eDM,acc:eR|srFPFlags|crFPMask||"vsub.f\\t$dst, $acc1, $acc2, $acc"
VSUB_f_vmac_cm2_add_scd|II_VSUB_f_vmac_cm2_add_scd|8||dst:eDM|acc1:eDM,c:mR31_scd,acc:eR|srFPFlags|crFPMask,crSCDEn||"vsub.f\\t$dst, $acc1, scd[r31], $acc"
VSUB_f_vmac_cm2_add_scd_incr|II_VSUB_f_vmac_cm2_add_scd_incr|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,acc:eR|srFPFlags|crFPMask,crSCDEn|r_out=r|"vsub.f\\t$dst, $acc1, scd[r31++], $acc"
VSUB_vmac_cm2_add_reg|II_VSUB_vmac_cm2_add_reg|0||dst:eDM|acc1:eDM,acc2:eDM,acc:eR||||"vsub\\t$dst, $acc1, $acc2, $acc"
VSUB_vmac_cm2_add_scd|II_VSUB_vmac_cm2_add_scd|8||dst:eDM|acc1:eDM,c:mR31_scd,acc:eR||crSCDEn||"vsub\\t$dst, $acc1, scd[r31], $acc"
VSUB_vmac_cm2_add_scd_incr|II_VSUB_vmac_cm2_add_scd_incr|8||dst:eDM,r_out:mR31_scd|acc1:eDM,r:mR31_scd,acc:eR||crSCDEn|r_out=r|"vsub\\t$dst, $acc1, scd[r31++], $acc"
VTANH|II_VTANH|0||dst:OP_mWm|src:OP_mBMm||||"vtanh\\t $dst, $src"
VUNPACK_mv_unpack_w_unpackSign0|II_VUNPACK_mv_unpack_w_unpackSign0|0||dst:OP_mXb|src:OP_mWs||crUnpackSize,unpackSign0||"vunpack\\t$dst, $src, unpacksign0"
VUNPACK_mv_unpack_w_unpackSign1|II_VUNPACK_mv_unpack_w_unpackSign1|0||dst:OP_mXb|src:OP_mWs||crUnpackSize,unpackSign1||"vunpack\\t$dst, $src, unpacksign1"
VUNPACK_mv_unpack_x_unpackSign0|II_VUNPACK_mv_unpack_x_unpackSign0|0||dst:eY|src:OP_mXs||crUnpackSize,unpackSign0||"vunpack\\t$dst, $src, unpacksign0"
VUNPACK_mv_unpack_x_unpackSign1|II_VUNPACK_mv_unpack_x_unpackSign1|0||dst:eY|src:OP_mXs||crUnpackSize,unpackSign1||"vunpack\\t$dst, $src, unpacksign1"
VUPS_2x_mv_ups_w2b_upsSign0|II_VUPS_2x_mv_ups_w2b_upsSign0|0||dst:OP_mBMm|src:OP_mWm,su:eS|srUPS_of|crSat,crUPSMode,upsSign0||"vups.2x\\t$dst, $src, $su, upssign0"
VUPS_2x_mv_ups_w2b_upsSign1|II_VUPS_2x_mv_ups_w2b_upsSign1|0||dst:OP_mBMm|src:OP_mWm,su:eS|srUPS_of|crSat,crUPSMode,upsSign1||"vups.2x\\t$dst, $src, $su, upssign1"
VUPS_2x_mv_ups_x2c_upsSign0|II_VUPS_2x_mv_ups_x2c_upsSign0|0||dst:OP_mCMm|src:OP_mXm,su:eS|srUPS_of|crSat,crUPSMode,upsSign0||"vups.2x\\t$dst, $src, $su, upssign0"
VUPS_2x_mv_ups_x2c_upsSign1|II_VUPS_2x_mv_ups_x2c_upsSign1|0||dst:OP_mCMm|src:OP_mXm,su:eS|srUPS_of|crSat,crUPSMode,upsSign1||"vups.2x\\t$dst, $src, $su, upssign1"
VUPS_4x_mv_ups_w2c_upsSign0|II_VUPS_4x_mv_ups_w2c_upsSign0|0||dst:OP_mCMm|src:OP_mWm,su:eS|srUPS_of|crSat,crUPSMode,upsSign0||"vups.4x\\t$dst, $src, $su, upssign0"
VUPS_4x_mv_ups_w2c_upsSign1|II_VUPS_4x_mv_ups_w2c_upsSign1|0||dst:OP_mCMm|src:OP_mWm,su:eS|srUPS_of|crSat,crUPSMode,upsSign1||"vups.4x\\t$dst, $src, $su, upssign1"
VUPS_4x_mv_ups_x2d_upsSign0|II_VUPS_4x_mv_ups_x2d_upsSign0|0||dst:eDM|src:OP_mXm,su:eS|srUPS_of|crSat,crUPSMode,upsSign0||"vups.4x\\t$dst, $src, $su, upssign0"
VUPS_4x_mv_ups_x2d_upsSign1|II_VUPS_4x_mv_ups_x2d_upsSign1|0||dst:eDM|src:OP_mXm,su:eS|srUPS_of|crSat,crUPSMode,upsSign1||"vups.4x\\t$dst, $src, $su, upssign1"
XOR|II_XOR|0||d0:eR|s0:eR,s1:eR||||"xor\\t $d0, $s0, $s1"
"""


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(value.split(",")) if value else ()


def _parse_physical_registers() -> tuple[PhysicalRegister, ...]:
    result = []
    for record in _PHYSICAL_REGISTER_RECORDS.splitlines():
        name, assembly_name, hardware_encoding, units, subregisters = record.split("|")
        subregister_pairs = tuple(
            pair.split(":", 1) for pair in _split_csv(subregisters)
        )
        result.append(
            PhysicalRegister(
                name=name,
                assembly_name=assembly_name,
                hardware_encoding=int(hardware_encoding),
                atomic_units=tuple(int(unit) for unit in units.split(",")),
                subregisters=tuple(pair[0] for pair in subregister_pairs),
                subregister_indices=tuple(pair[1] for pair in subregister_pairs),
            )
        )
    return tuple(result)


_REGISTER_LAYOUTS = tuple(
    RegisterLayout(*record) for record in _REGISTER_LAYOUT_RECORDS
)


def _parse_register_classes() -> tuple[RegisterClass, ...]:
    result = []
    for record in _REGISTER_CLASS_RECORDS.splitlines():
        name, layout_id, value_types_id, flags, candidates = record.split("|")
        flag_bits = int(flags)
        result.append(
            RegisterClass(
                name=name,
                layout=_REGISTER_LAYOUTS[int(layout_id)],
                value_types=_VALUE_TYPE_GROUPS[int(value_types_id)],
                candidates=_split_csv(candidates),
                is_allocatable=bool(flag_bits & 1),
                consider_in_pre_ra_scheduling=bool(flag_bits & 2),
                generate_pressure_set=bool(flag_bits & 4),
            )
        )
    return tuple(result)


def _derive_register_classes(
    source_classes: tuple[RegisterClass, ...],
    physical_registers: tuple[PhysicalRegister, ...],
) -> tuple[RegisterClass, ...]:
    """Derives Loom storage domains that intersect native operand classes."""

    classes = {register_class.name: register_class for register_class in source_classes}
    registers = {register.name: register for register in physical_registers}
    el_class = classes["eL"]
    native_short_predicates = set(classes["eRS16"].candidates)
    predicate_candidates = tuple(
        register_name
        for register_name in el_class.candidates
        if registers[register_name].subregisters[0] in native_short_predicates
    )
    if predicate_candidates != tuple(f"l{index}" for index in range(8, 16)):
        raise ValueError(
            "AIE2P cross-width predicate intersection is not l8 through l15"
        )
    return (
        RegisterClass(
            name="eLPredicate",
            layout=el_class.layout,
            value_types=el_class.value_types,
            candidates=predicate_candidates,
            is_allocatable=el_class.is_allocatable,
            consider_in_pre_ra_scheduling=el_class.consider_in_pre_ra_scheduling,
            generate_pressure_set=el_class.generate_pressure_set,
        ),
    )


_REGISTER_ENCODING_MAPS = tuple(
    tuple(
        (register, int(value))
        for register, value in (pair.split(":") for pair in record.split(","))
    )
    for record in _REGISTER_ENCODING_MAP_RECORDS
)


def _parse_register_adapters() -> tuple[RegisterAdapter, ...]:
    return tuple(
        RegisterAdapter(
            name=name,
            register_class=register_class,
            register_encodings=_REGISTER_ENCODING_MAPS[int(map_id)],
            architectural_encodings=(
                _REGISTER_ENCODING_MAPS[int(architectural_map_id)]
                if architectural_map_id
                else ()
            ),
        )
        for name, register_class, map_id, architectural_map_id in (
            record.split("|") for record in _REGISTER_ADAPTER_RECORDS.splitlines()
        )
    )


def _derive_el_subregister_adapters(
    source_adapters: tuple[RegisterAdapter, ...],
    register_classes: tuple[RegisterClass, ...],
    physical_registers: tuple[PhysicalRegister, ...],
) -> tuple[RegisterAdapter, ...]:
    """Projects each cross-width predicate register onto its scalar halves.

    These adapters are Loom-owned derivatives of the physical subregister table
    above, not additional llvm-aie source records. The direct pair encodes
    scalar-register fields by architectural register number. The mLdaCg variant
    composes the high-half projection with that operand's source adapter. This
    lets descriptors use one allocatable predicate value with either encoding
    domain.
    """

    registers = {register.name: register for register in physical_registers}
    adapters = {adapter.name: adapter for adapter in source_adapters}
    el_class = next(row for row in register_classes if row.name == "eLPredicate")
    expected_subregister_indices = ("sub_l_even", "sub_l_odd")
    projected_maps: list[list[tuple[str, int]]] = [[], []]
    lda_values = dict(adapters["OP_mLdaCg"].effective_register_encodings)
    lda_high_map: list[tuple[str, int]] = []
    for register_name in el_class.candidates:
        register = registers[register_name]
        if register.subregister_indices != expected_subregister_indices:
            raise ValueError(
                f"{register_name}: eL scalar subregister order is "
                f"{register.subregister_indices}, expected "
                f"{expected_subregister_indices}"
            )
        for projected_map, subregister_name in zip(
            projected_maps, register.subregisters, strict=True
        ):
            projected_map.append(
                (register_name, registers[subregister_name].hardware_encoding)
            )
        lda_high_map.append((register_name, lda_values[register.subregisters[1]]))
    return (
        RegisterAdapter(
            name="LOOM_eL_low32",
            register_class="eLPredicate",
            register_encodings=tuple(projected_maps[0]),
        ),
        RegisterAdapter(
            name="LOOM_eL_high32",
            register_class="eLPredicate",
            register_encodings=tuple(projected_maps[1]),
        ),
        RegisterAdapter(
            name="LOOM_eL_high32_OP_mLdaCg",
            register_class="eLPredicate",
            register_encodings=tuple(lda_high_map),
        ),
    )


def _derive_vector_storage_adapters(
    source_adapters: tuple[RegisterAdapter, ...],
    register_classes: tuple[RegisterClass, ...],
    physical_registers: tuple[PhysicalRegister, ...],
) -> tuple[RegisterAdapter, ...]:
    """Derives exact operand encodings for Loom vector storage domains.

    A 128-bit BF16 outer-product operand occupies the low half of one W
    register. Loads and stores therefore need the native W operand encodings
    restricted to eWL, while VEXTBCST needs each eWL candidate encoded as its
    containing X register. VMOV's source classes include X, accumulator, FIFO,
    and state registers; the final two adapters retain only its X domain so a
    descriptor can expose the result as ordinary VEC256 storage.
    """

    classes = {
        register_class.name: register_class for register_class in register_classes
    }
    registers = {register.name: register for register in physical_registers}
    adapters = {adapter.name: adapter for adapter in source_adapters}

    def restrict_adapter(
        name: str, source_name: str, register_class_name: str
    ) -> RegisterAdapter:
        source_values = dict(adapters[source_name].effective_register_encodings)
        candidates = classes[register_class_name].candidates
        missing = set(candidates) - set(source_values)
        if missing:
            raise ValueError(
                f"{source_name}: cannot derive {name}; missing registers {sorted(missing)}"
            )
        return RegisterAdapter(
            name=name,
            register_class=register_class_name,
            register_encodings=tuple(
                (register_name, source_values[register_name])
                for register_name in candidates
            ),
        )

    xm_values = dict(adapters["OP_mXm"].effective_register_encodings)
    ewl_candidates = classes["eWL"].candidates
    ewl_as_x: dict[str, int] = {}
    for x_name in classes["mXm"].candidates:
        x_register = registers[x_name]
        if x_register.subregister_indices != ("sub_256_lo", "sub_256_hi"):
            raise ValueError(
                f"{x_name}: X register has unexpected subregister order "
                f"{x_register.subregister_indices}"
            )
        low_w_name = x_register.subregisters[0]
        if low_w_name in ewl_candidates:
            ewl_as_x[low_w_name] = xm_values[x_name]
    if set(ewl_as_x) != set(ewl_candidates):
        raise ValueError("AIE2P eWL candidates do not map exactly onto X low halves")

    return (
        restrict_adapter("LOOM_eWL_OP_mWa", "OP_mWa", "eWL"),
        restrict_adapter("LOOM_eWL_OP_mWb", "OP_mWb", "eWL"),
        restrict_adapter("LOOM_eWL_OP_mWs", "OP_mWs", "eWL"),
        RegisterAdapter(
            name="LOOM_eWL_OP_mXm",
            register_class="eWL",
            register_encodings=tuple(
                (register_name, ewl_as_x[register_name])
                for register_name in ewl_candidates
            ),
        ),
        restrict_adapter("LOOM_mXm_OP_mMvBMXDst", "OP_mMvBMXDst", "mXm"),
        restrict_adapter("LOOM_mXm_OP_mMvBMXSrc", "OP_mMvBMXSrc", "mXm"),
    )


def _parse_immediates() -> tuple[ImmediateEncoding, ...]:
    result = []
    for record in _IMMEDIATE_ENCODING_RECORDS.splitlines():
        name, semantic_width, encoded_width, step, flags = record.split("|")
        flag_bits = int(flags)
        result.append(
            ImmediateEncoding(
                name=name,
                semantic_width_bits=int(semantic_width),
                encoded_width_bits=int(encoded_width),
                step=int(step),
                is_signed=bool(flag_bits & 1),
                is_negative=bool(flag_bits & 2),
                allows_symbol_reference=bool(flag_bits & 4),
            )
        )
    return tuple(result)


_PHYSICAL_REGISTERS = _parse_physical_registers()
_SOURCE_REGISTER_CLASSES = _parse_register_classes()
_REGISTER_CLASSES = (
    *_SOURCE_REGISTER_CLASSES,
    *_derive_register_classes(_SOURCE_REGISTER_CLASSES, _PHYSICAL_REGISTERS),
)
_SOURCE_REGISTER_ADAPTERS = _parse_register_adapters()
_REGISTER_ADAPTERS = (
    *_SOURCE_REGISTER_ADAPTERS,
    *_derive_el_subregister_adapters(
        _SOURCE_REGISTER_ADAPTERS,
        _REGISTER_CLASSES,
        _PHYSICAL_REGISTERS,
    ),
    *_derive_vector_storage_adapters(
        _SOURCE_REGISTER_ADAPTERS,
        _REGISTER_CLASSES,
        _PHYSICAL_REGISTERS,
    ),
)
_IMMEDIATES = _parse_immediates()
_REGISTER_CLASS_NAMES = {row.name for row in _REGISTER_CLASSES}
_REGISTER_ADAPTER_NAMES = {row.name for row in _REGISTER_ADAPTERS}
_IMMEDIATE_NAMES = {row.name for row in _IMMEDIATES}


def _parse_operand(record: str) -> MachineOperand:
    name, type_name = record.split(":")
    if type_name in _REGISTER_CLASS_NAMES:
        kind = MachineOperandKind.REGISTER_CLASS
    elif type_name in _REGISTER_ADAPTER_NAMES:
        kind = MachineOperandKind.REGISTER_ADAPTER
    elif type_name in _IMMEDIATE_NAMES:
        kind = MachineOperandKind.IMMEDIATE
    else:
        raise ValueError(f"{name}: unknown AIE2P operand type {type_name}")
    return MachineOperand(name=name, type_name=type_name, kind=kind)


def _parse_operands(record: str) -> tuple[MachineOperand, ...]:
    return tuple(_parse_operand(value) for value in _split_csv(record))


def _parse_forms() -> tuple[MachineForm, ...]:
    result = []
    for record in _MACHINE_FORM_RECORDS.splitlines():
        (
            name,
            itinerary,
            properties,
            control_flow_kind,
            outputs,
            inputs,
            implicit_defs,
            implicit_uses,
            ties,
            assembly,
        ) = record.split("|", 9)
        result.append(
            MachineForm(
                name=name,
                itinerary=itinerary,
                property_bits=int(properties, 16),
                control_flow_kind=control_flow_kind or None,
                outputs=_parse_operands(outputs),
                inputs=_parse_operands(inputs),
                implicit_defs=_split_csv(implicit_defs),
                implicit_uses=_split_csv(implicit_uses),
                ties=tuple(MachineTie(*tie.split("=", 1)) for tie in _split_csv(ties)),
                assembly=json.loads(assembly),
            )
        )
    return tuple(result)


CORE_MACHINE_TABLE = MachineTable(
    atomic_unit_names=tuple(_ATOMIC_UNIT_NAME_RECORDS.splitlines()),
    physical_registers=_PHYSICAL_REGISTERS,
    register_classes=_REGISTER_CLASSES,
    register_adapters=_REGISTER_ADAPTERS,
    immediates=_IMMEDIATES,
    forms=_parse_forms(),
)

validate_machine_table(CORE_MACHINE_TABLE, CORE_ENCODING_TABLE)
