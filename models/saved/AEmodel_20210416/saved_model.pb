??)
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18ͥ#
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?f*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?f*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:f*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:f*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
l
z/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
z/kernel
e
z/kernel/Read/ReadVariableOpReadVariableOpz/kernel*
_output_shapes

:*
dtype0
d
z/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez/bias
]
z/bias/Read/ReadVariableOpReadVariableOpz/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:f*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:f*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f?*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	f?*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:?*
dtype0
?
(conv1d_transpose/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(conv1d_transpose/conv2d_transpose/kernel
?
<conv1d_transpose/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp(conv1d_transpose/conv2d_transpose/kernel*&
_output_shapes
:*
dtype0
?
&conv1d_transpose/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&conv1d_transpose/conv2d_transpose/bias
?
:conv1d_transpose/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp&conv1d_transpose/conv2d_transpose/bias*
_output_shapes
:*
dtype0
?
,conv1d_transpose_1/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,conv1d_transpose_1/conv2d_transpose_1/kernel
?
@conv1d_transpose_1/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp,conv1d_transpose_1/conv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0
?
*conv1d_transpose_1/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*conv1d_transpose_1/conv2d_transpose_1/bias
?
>conv1d_transpose_1/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp*conv1d_transpose_1/conv2d_transpose_1/bias*
_output_shapes
:*
dtype0
?
conv_2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_2d_transpose/kernel
?
,conv_2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv_2d_transpose/kernel*&
_output_shapes
:*
dtype0
?
conv_2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv_2d_transpose/bias
}
*conv_2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv_2d_transpose/bias*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d/kernel/m
?
(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_1/kernel/m
?
*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
:*
dtype0
?
Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?f*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?f*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:f*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:f*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
z
Adam/z/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nameAdam/z/kernel/m
s
#Adam/z/kernel/m/Read/ReadVariableOpReadVariableOpAdam/z/kernel/m*
_output_shapes

:*
dtype0
r
Adam/z/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/z/bias/m
k
!Adam/z/bias/m/Read/ReadVariableOpReadVariableOpAdam/z/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:f*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:f*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f?*&
shared_nameAdam/dense_4/kernel/m
?
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	f?*
dtype0

Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_4/bias/m
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes	
:?*
dtype0
?
/Adam/conv1d_transpose/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/conv1d_transpose/conv2d_transpose/kernel/m
?
CAdam/conv1d_transpose/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/conv1d_transpose/conv2d_transpose/kernel/m*&
_output_shapes
:*
dtype0
?
-Adam/conv1d_transpose/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/conv1d_transpose/conv2d_transpose/bias/m
?
AAdam/conv1d_transpose/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOp-Adam/conv1d_transpose/conv2d_transpose/bias/m*
_output_shapes
:*
dtype0
?
3Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/m
?
GAdam/conv1d_transpose_1/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp3Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/m*&
_output_shapes
:*
dtype0
?
1Adam/conv1d_transpose_1/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/conv1d_transpose_1/conv2d_transpose_1/bias/m
?
EAdam/conv1d_transpose_1/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOp1Adam/conv1d_transpose_1/conv2d_transpose_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv_2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv_2d_transpose/kernel/m
?
3Adam/conv_2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_2d_transpose/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv_2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/conv_2d_transpose/bias/m
?
1Adam/conv_2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_2d_transpose/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d/kernel/v
?
(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_1/kernel/v
?
*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
:*
dtype0
?
Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?f*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?f*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:f*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:f*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
z
Adam/z/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nameAdam/z/kernel/v
s
#Adam/z/kernel/v/Read/ReadVariableOpReadVariableOpAdam/z/kernel/v*
_output_shapes

:*
dtype0
r
Adam/z/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/z/bias/v
k
!Adam/z/bias/v/Read/ReadVariableOpReadVariableOpAdam/z/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:f*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:f*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f?*&
shared_nameAdam/dense_4/kernel/v
?
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	f?*
dtype0

Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_4/bias/v
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes	
:?*
dtype0
?
/Adam/conv1d_transpose/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/conv1d_transpose/conv2d_transpose/kernel/v
?
CAdam/conv1d_transpose/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/conv1d_transpose/conv2d_transpose/kernel/v*&
_output_shapes
:*
dtype0
?
-Adam/conv1d_transpose/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/conv1d_transpose/conv2d_transpose/bias/v
?
AAdam/conv1d_transpose/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOp-Adam/conv1d_transpose/conv2d_transpose/bias/v*
_output_shapes
:*
dtype0
?
3Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/v
?
GAdam/conv1d_transpose_1/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp3Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/v*&
_output_shapes
:*
dtype0
?
1Adam/conv1d_transpose_1/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/conv1d_transpose_1/conv2d_transpose_1/bias/v
?
EAdam/conv1d_transpose_1/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOp1Adam/conv1d_transpose_1/conv2d_transpose_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv_2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv_2d_transpose/kernel/v
?
3Adam/conv_2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_2d_transpose/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv_2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/conv_2d_transpose/bias/v
?
1Adam/conv_2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_2d_transpose/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*۟
valueПB̟ Bğ
?
shape_convolved
encoder
decoder
	optimizer
loss
trainable_variables
regularization_losses
	variables
		keras_api


signatures
 
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer_with_weights-3
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
trainable_variables
regularization_losses
	variables
	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
 layer-5
!layer_with_weights-3
!layer-6
"layer_with_weights-4
"layer-7
#layer-8
$layer_with_weights-5
$layer-9
%layer-10
&layer-11
'trainable_variables
(regularization_losses
)	variables
*	keras_api
?
+iter

,beta_1

-beta_2
	.decay
/learning_rate0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?
 
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
 
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
?
trainable_variables

Hlayers
Ilayer_metrics
Jnon_trainable_variables
Kmetrics
Llayer_regularization_losses
regularization_losses
	variables
 
 
R
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
R
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
h

0kernel
1bias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
R
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
h

2kernel
3bias
]trainable_variables
^regularization_losses
_	variables
`	keras_api
h

4kernel
5bias
atrainable_variables
bregularization_losses
c	variables
d	keras_api
R
etrainable_variables
fregularization_losses
g	variables
h	keras_api
R
itrainable_variables
jregularization_losses
k	variables
l	keras_api
h

6kernel
7bias
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
h

8kernel
9bias
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
h

:kernel
;bias
utrainable_variables
vregularization_losses
w	variables
x	keras_api
V
00
11
22
33
44
55
66
77
88
99
:10
;11
 
V
00
11
22
33
44
55
66
77
88
99
:10
;11
?
trainable_variables

ylayers
zlayer_metrics
{non_trainable_variables
|metrics
}layer_regularization_losses
regularization_losses
	variables
 
j

<kernel
=bias
~trainable_variables
regularization_losses
?	variables
?	keras_api
l

>kernel
?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

@kernel
Abias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?ExpandChannel
?ConvTranspose
?SqueezeChannel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?ExpandChannel
?ConvTranspose
?SqueezeChannel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

Fkernel
Gbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
 
V
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
?
'trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
(regularization_losses
)	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv1d/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv1d/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv1d_1/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEz/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEz/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_2/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_2/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_3/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_3/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_4/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_4/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(conv1d_transpose/conv2d_transpose/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE&conv1d_transpose/conv2d_transpose/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,conv1d_transpose_1/conv2d_transpose_1/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*conv1d_transpose_1/conv2d_transpose_1/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv_2d_transpose/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv_2d_transpose/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
 
 
 
 
 
?
Mtrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
Nregularization_losses
O	variables
 
 
 
?
Qtrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
Rregularization_losses
S	variables

00
11
 

00
11
?
Utrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
Vregularization_losses
W	variables
 
 
 
?
Ytrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
Zregularization_losses
[	variables

20
31
 

20
31
?
]trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
^regularization_losses
_	variables

40
51
 

40
51
?
atrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
bregularization_losses
c	variables
 
 
 
?
etrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
fregularization_losses
g	variables
 
 
 
?
itrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
jregularization_losses
k	variables

60
71
 

60
71
?
mtrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
nregularization_losses
o	variables

80
91
 

80
91
?
qtrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
rregularization_losses
s	variables

:0
;1
 

:0
;1
?
utrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
vregularization_losses
w	variables
V
0
1
2
3
4
5
6
7
8
9
10
11
 
 
 
 

<0
=1
 

<0
=1
?
~trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
regularization_losses
?	variables

>0
?1
 

>0
?1
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables

@0
A1
 

@0
A1
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

Bkernel
Cbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api

B0
C1
 

B0
C1
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

Dkernel
Ebias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api

D0
E1
 

D0
E1
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables

F0
G1
 

F0
G1
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
V
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables

B0
C1
 

B0
C1
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables

?0
?1
?2
 
 
 
 
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables

D0
E1
 

D0
E1
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
 
 
 
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables

?0
?1
?2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
vt
VARIABLE_VALUEAdam/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv1d/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d_1/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d_1/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/z/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/z/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_2/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_2/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_3/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_3/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_4/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_4/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/conv1d_transpose/conv2d_transpose/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/conv1d_transpose/conv2d_transpose/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/conv1d_transpose_1/conv2d_transpose_1/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_2d_transpose/kernel/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv_2d_transpose/bias/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv1d/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d_1/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d_1/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/z/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/z/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_2/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_2/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_3/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_3/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_4/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_4/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/conv1d_transpose/conv2d_transpose/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/conv1d_transpose/conv2d_transpose/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/conv1d_transpose_1/conv2d_transpose_1/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_2d_transpose/kernel/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv_2d_transpose/bias/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????d*
dtype0* 
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasz/kernelz/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias(conv1d_transpose/conv2d_transpose/kernel&conv1d_transpose/conv2d_transpose/bias,conv1d_transpose_1/conv2d_transpose_1/kernel*conv1d_transpose_1/conv2d_transpose_1/biasconv_2d_transpose/kernelconv_2d_transpose/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_4697948
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpz/kernel/Read/ReadVariableOpz/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp<conv1d_transpose/conv2d_transpose/kernel/Read/ReadVariableOp:conv1d_transpose/conv2d_transpose/bias/Read/ReadVariableOp@conv1d_transpose_1/conv2d_transpose_1/kernel/Read/ReadVariableOp>conv1d_transpose_1/conv2d_transpose_1/bias/Read/ReadVariableOp,conv_2d_transpose/kernel/Read/ReadVariableOp*conv_2d_transpose/bias/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp#Adam/z/kernel/m/Read/ReadVariableOp!Adam/z/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOpCAdam/conv1d_transpose/conv2d_transpose/kernel/m/Read/ReadVariableOpAAdam/conv1d_transpose/conv2d_transpose/bias/m/Read/ReadVariableOpGAdam/conv1d_transpose_1/conv2d_transpose_1/kernel/m/Read/ReadVariableOpEAdam/conv1d_transpose_1/conv2d_transpose_1/bias/m/Read/ReadVariableOp3Adam/conv_2d_transpose/kernel/m/Read/ReadVariableOp1Adam/conv_2d_transpose/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp#Adam/z/kernel/v/Read/ReadVariableOp!Adam/z/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpCAdam/conv1d_transpose/conv2d_transpose/kernel/v/Read/ReadVariableOpAAdam/conv1d_transpose/conv2d_transpose/bias/v/Read/ReadVariableOpGAdam/conv1d_transpose_1/conv2d_transpose_1/kernel/v/Read/ReadVariableOpEAdam/conv1d_transpose_1/conv2d_transpose_1/bias/v/Read/ReadVariableOp3Adam/conv_2d_transpose/kernel/v/Read/ReadVariableOp1Adam/conv_2d_transpose/bias/v/Read/ReadVariableOpConst*Z
TinS
Q2O	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_4700385
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasz/kernelz/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias(conv1d_transpose/conv2d_transpose/kernel&conv1d_transpose/conv2d_transpose/bias,conv1d_transpose_1/conv2d_transpose_1/kernel*conv1d_transpose_1/conv2d_transpose_1/biasconv_2d_transpose/kernelconv_2d_transpose/biasAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/z/kernel/mAdam/z/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/m/Adam/conv1d_transpose/conv2d_transpose/kernel/m-Adam/conv1d_transpose/conv2d_transpose/bias/m3Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/m1Adam/conv1d_transpose_1/conv2d_transpose_1/bias/mAdam/conv_2d_transpose/kernel/mAdam/conv_2d_transpose/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/z/kernel/vAdam/z/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/v/Adam/conv1d_transpose/conv2d_transpose/kernel/v-Adam/conv1d_transpose/conv2d_transpose/bias/v3Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/v1Adam/conv1d_transpose_1/conv2d_transpose_1/bias/vAdam/conv_2d_transpose/kernel/vAdam/conv_2d_transpose/bias/v*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_4700626?? 
?
a
E__inference_lambda_7_layer_call_and_return_conditional_losses_4697293

inputs
identity?
SqueezeSqueezeinputs*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?.
?
D__inference_encoder_layer_call_and_return_conditional_losses_4696619
encoder_input
conv2d_4696431
conv2d_4696433
conv1d_4696486
conv1d_4696488
conv1d_1_4696518
conv1d_1_4696520
dense_4696560
dense_4696562
dense_1_4696587
dense_1_4696589
	z_4696613
	z_4696615
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?z/StatefulPartitionedCall?
Std_Normalize/PartitionedCallPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_46963762
Std_Normalize/PartitionedCall?
lambda/PartitionedCallPartitionedCall&Std_Normalize/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_46963902
lambda/PartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_4696431conv2d_4696433*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_46964202 
conv2d/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_46964412
lambda_1/PartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_4696486conv1d_4696488*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_46964752 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_4696518conv1d_1_4696520*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_46965072"
 conv1d_1/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_layer_call_and_return_conditional_losses_46963582#
!average_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall*average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_46965302
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4696560dense_4696562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_46965492
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4696587dense_1_4696589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_46965762!
dense_1/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0	z_4696613	z_4696615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_z_layer_call_and_return_conditional_losses_46966022
z/StatefulPartitionedCall?
IdentityIdentity"z/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????d
'
_user_specified_nameencoder_input
?
O
3__inference_average_pooling1d_layer_call_fn_4696364

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_layer_call_and_return_conditional_losses_46963582
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

*__inference_conv1d_1_layer_call_fn_4699756

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_46965072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????`::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_layer_call_and_return_conditional_losses_4696420

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:?????????b2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d:::W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
|
'__inference_dense_layer_call_fn_4699787

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_46965492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_4699691

inputs
identityr
SqueezeSqueezeinputs*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????b:W S
/
_output_shapes
:?????????b
 
_user_specified_nameinputs
?
?
D__inference_dense_3_layer_call_and_return_conditional_losses_4697002

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_4699696

inputs
identityr
SqueezeSqueezeinputs*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????b:W S
/
_output_shapes
:?????????b
 
_user_specified_nameinputs
?
~
)__inference_dense_1_layer_call_fn_4699807

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_46965762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????f::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?	
?
)__inference_decoder_layer_call_fn_4697499
z
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46974722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namez
?
?
6__inference_particle_autoencoder_layer_call_fn_4698496
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_46977812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_4699899

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????/2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????/2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
4__inference_conv1d_transpose_1_layer_call_fn_4700076

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_46972282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv_2d_transpose_layer_call_fn_4696960

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv_2d_transpose_layer_call_and_return_conditional_losses_46969502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_4700082

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2

ExpandDimsx
IdentityIdentityExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
a
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_4699639
x
identityc
sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
sub/yZ
subSubxsub/y:output:0*
T0*+
_output_shapes
:?????????d2
subk
	truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
	truediv/yp
truedivRealDivsub:z:0truediv/y:output:0*
T0*+
_output_shapes
:?????????d2	
truedivc
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:N J
+
_output_shapes
:?????????d

_user_specified_namex
ۨ
?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698169
input_11
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource>
:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource2
.encoder_conv1d_biasadd_readvariableop_resource@
<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_1_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.encoder_dense_1_matmul_readvariableop_resource3
/encoder_dense_1_biasadd_readvariableop_resource,
(encoder_z_matmul_readvariableop_resource-
)encoder_z_biasadd_readvariableop_resource2
.decoder_dense_2_matmul_readvariableop_resource3
/decoder_dense_2_biasadd_readvariableop_resource2
.decoder_dense_3_matmul_readvariableop_resource3
/decoder_dense_3_biasadd_readvariableop_resource2
.decoder_dense_4_matmul_readvariableop_resource3
/decoder_dense_4_biasadd_readvariableop_resourceV
Rdecoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resourceM
Idecoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resourceZ
Vdecoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceQ
Mdecoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resourceF
Bdecoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource=
9decoder_conv_2d_transpose_biasadd_readvariableop_resource
identity??
encoder/Std_Normalize/sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
encoder/Std_Normalize/sub/y?
encoder/Std_Normalize/subSubinput_1$encoder/Std_Normalize/sub/y:output:0*
T0*+
_output_shapes
:?????????d2
encoder/Std_Normalize/sub?
encoder/Std_Normalize/truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2!
encoder/Std_Normalize/truediv/y?
encoder/Std_Normalize/truedivRealDivencoder/Std_Normalize/sub:z:0(encoder/Std_Normalize/truediv/y:output:0*
T0*+
_output_shapes
:?????????d2
encoder/Std_Normalize/truediv?
encoder/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
encoder/lambda/ExpandDims/dim?
encoder/lambda/ExpandDims
ExpandDims!encoder/Std_Normalize/truediv:z:0&encoder/lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2
encoder/lambda/ExpandDims?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2D"encoder/lambda/ExpandDims:output:0,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2
encoder/conv2d/BiasAdd?
encoder/conv2d/EluEluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2
encoder/conv2d/Elu?
encoder/lambda_1/SqueezeSqueeze encoder/conv2d/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2
encoder/lambda_1/Squeeze?
$encoder/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$encoder/conv1d/conv1d/ExpandDims/dim?
 encoder/conv1d/conv1d/ExpandDims
ExpandDims!encoder/lambda_1/Squeeze:output:0-encoder/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2"
 encoder/conv1d/conv1d/ExpandDims?
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype023
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
&encoder/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&encoder/conv1d/conv1d/ExpandDims_1/dim?
"encoder/conv1d/conv1d/ExpandDims_1
ExpandDims9encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0/encoder/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2$
"encoder/conv1d/conv1d/ExpandDims_1?
encoder/conv1d/conv1dConv2D)encoder/conv1d/conv1d/ExpandDims:output:0+encoder/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2
encoder/conv1d/conv1d?
encoder/conv1d/conv1d/SqueezeSqueezeencoder/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????2
encoder/conv1d/conv1d/Squeeze?
%encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv1d/BiasAdd/ReadVariableOp?
encoder/conv1d/BiasAddBiasAdd&encoder/conv1d/conv1d/Squeeze:output:0-encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
encoder/conv1d/BiasAdd?
encoder/conv1d/EluEluencoder/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
encoder/conv1d/Elu?
&encoder/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&encoder/conv1d_1/conv1d/ExpandDims/dim?
"encoder/conv1d_1/conv1d/ExpandDims
ExpandDims encoder/conv1d/Elu:activations:0/encoder/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2$
"encoder/conv1d_1/conv1d/ExpandDims?
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype025
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
(encoder/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_1/conv1d/ExpandDims_1/dim?
$encoder/conv1d_1/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2&
$encoder/conv1d_1/conv1d/ExpandDims_1?
encoder/conv1d_1/conv1dConv2D+encoder/conv1d_1/conv1d/ExpandDims:output:0-encoder/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2
encoder/conv1d_1/conv1d?
encoder/conv1d_1/conv1d/SqueezeSqueeze encoder/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????2!
encoder/conv1d_1/conv1d/Squeeze?
'encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'encoder/conv1d_1/BiasAdd/ReadVariableOp?
encoder/conv1d_1/BiasAddBiasAdd(encoder/conv1d_1/conv1d/Squeeze:output:0/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2
encoder/conv1d_1/BiasAdd?
encoder/conv1d_1/EluElu!encoder/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^2
encoder/conv1d_1/Elu?
(encoder/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(encoder/average_pooling1d/ExpandDims/dim?
$encoder/average_pooling1d/ExpandDims
ExpandDims"encoder/conv1d_1/Elu:activations:01encoder/average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2&
$encoder/average_pooling1d/ExpandDims?
!encoder/average_pooling1d/AvgPoolAvgPool-encoder/average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2#
!encoder/average_pooling1d/AvgPool?
!encoder/average_pooling1d/SqueezeSqueeze*encoder/average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2#
!encoder/average_pooling1d/Squeeze
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape*encoder/average_pooling1d/Squeeze:output:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/BiasAdd
encoder/dense/EluEluencoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/Elu?
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02'
%encoder/dense_1/MatMul/ReadVariableOp?
encoder/dense_1/MatMulMatMulencoder/dense/Elu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/MatMul?
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_1/BiasAdd/ReadVariableOp?
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/BiasAdd?
encoder/dense_1/EluElu encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/Elu?
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
encoder/z/MatMul/ReadVariableOp?
encoder/z/MatMulMatMul!encoder/dense_1/Elu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z/MatMul?
 encoder/z/BiasAdd/ReadVariableOpReadVariableOp)encoder_z_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 encoder/z/BiasAdd/ReadVariableOp?
encoder/z/BiasAddBiasAddencoder/z/MatMul:product:0(encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z/BiasAdd?
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%decoder/dense_2/MatMul/ReadVariableOp?
decoder/dense_2/MatMulMatMulencoder/z/BiasAdd:output:0-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/MatMul?
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&decoder/dense_2/BiasAdd/ReadVariableOp?
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/BiasAdd?
decoder/dense_2/EluElu decoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/Elu?
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02'
%decoder/dense_3/MatMul/ReadVariableOp?
decoder/dense_3/MatMulMatMul!decoder/dense_2/Elu:activations:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/MatMul?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02(
&decoder/dense_3/BiasAdd/ReadVariableOp?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/BiasAdd?
decoder/dense_3/EluElu decoder/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/Elu?
%decoder/dense_4/MatMul/ReadVariableOpReadVariableOp.decoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02'
%decoder/dense_4/MatMul/ReadVariableOp?
decoder/dense_4/MatMulMatMul!decoder/dense_3/Elu:activations:0-decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/MatMul?
&decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_4/BiasAdd/ReadVariableOp?
decoder/dense_4/BiasAddBiasAdd decoder/dense_4/MatMul:product:0.decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/BiasAdd?
decoder/dense_4/EluElu decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/Elu
decoder/reshape/ShapeShape!decoder/dense_4/Elu:activations:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshape!decoder/dense_4/Elu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????/2
decoder/reshape/Reshape|
decoder/up_sampling1d/ConstConst*
_output_shapes
: *
dtype0*
value	B :/2
decoder/up_sampling1d/Const?
%decoder/up_sampling1d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%decoder/up_sampling1d/split/split_dim?

decoder/up_sampling1d/splitSplit.decoder/up_sampling1d/split/split_dim:output:0 decoder/reshape/Reshape:output:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split/2
decoder/up_sampling1d/split?
!decoder/up_sampling1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/up_sampling1d/concat/axis?
decoder/up_sampling1d/concatConcatV2$decoder/up_sampling1d/split:output:0$decoder/up_sampling1d/split:output:0$decoder/up_sampling1d/split:output:1$decoder/up_sampling1d/split:output:1$decoder/up_sampling1d/split:output:2$decoder/up_sampling1d/split:output:2$decoder/up_sampling1d/split:output:3$decoder/up_sampling1d/split:output:3$decoder/up_sampling1d/split:output:4$decoder/up_sampling1d/split:output:4$decoder/up_sampling1d/split:output:5$decoder/up_sampling1d/split:output:5$decoder/up_sampling1d/split:output:6$decoder/up_sampling1d/split:output:6$decoder/up_sampling1d/split:output:7$decoder/up_sampling1d/split:output:7$decoder/up_sampling1d/split:output:8$decoder/up_sampling1d/split:output:8$decoder/up_sampling1d/split:output:9$decoder/up_sampling1d/split:output:9%decoder/up_sampling1d/split:output:10%decoder/up_sampling1d/split:output:10%decoder/up_sampling1d/split:output:11%decoder/up_sampling1d/split:output:11%decoder/up_sampling1d/split:output:12%decoder/up_sampling1d/split:output:12%decoder/up_sampling1d/split:output:13%decoder/up_sampling1d/split:output:13%decoder/up_sampling1d/split:output:14%decoder/up_sampling1d/split:output:14%decoder/up_sampling1d/split:output:15%decoder/up_sampling1d/split:output:15%decoder/up_sampling1d/split:output:16%decoder/up_sampling1d/split:output:16%decoder/up_sampling1d/split:output:17%decoder/up_sampling1d/split:output:17%decoder/up_sampling1d/split:output:18%decoder/up_sampling1d/split:output:18%decoder/up_sampling1d/split:output:19%decoder/up_sampling1d/split:output:19%decoder/up_sampling1d/split:output:20%decoder/up_sampling1d/split:output:20%decoder/up_sampling1d/split:output:21%decoder/up_sampling1d/split:output:21%decoder/up_sampling1d/split:output:22%decoder/up_sampling1d/split:output:22%decoder/up_sampling1d/split:output:23%decoder/up_sampling1d/split:output:23%decoder/up_sampling1d/split:output:24%decoder/up_sampling1d/split:output:24%decoder/up_sampling1d/split:output:25%decoder/up_sampling1d/split:output:25%decoder/up_sampling1d/split:output:26%decoder/up_sampling1d/split:output:26%decoder/up_sampling1d/split:output:27%decoder/up_sampling1d/split:output:27%decoder/up_sampling1d/split:output:28%decoder/up_sampling1d/split:output:28%decoder/up_sampling1d/split:output:29%decoder/up_sampling1d/split:output:29%decoder/up_sampling1d/split:output:30%decoder/up_sampling1d/split:output:30%decoder/up_sampling1d/split:output:31%decoder/up_sampling1d/split:output:31%decoder/up_sampling1d/split:output:32%decoder/up_sampling1d/split:output:32%decoder/up_sampling1d/split:output:33%decoder/up_sampling1d/split:output:33%decoder/up_sampling1d/split:output:34%decoder/up_sampling1d/split:output:34%decoder/up_sampling1d/split:output:35%decoder/up_sampling1d/split:output:35%decoder/up_sampling1d/split:output:36%decoder/up_sampling1d/split:output:36%decoder/up_sampling1d/split:output:37%decoder/up_sampling1d/split:output:37%decoder/up_sampling1d/split:output:38%decoder/up_sampling1d/split:output:38%decoder/up_sampling1d/split:output:39%decoder/up_sampling1d/split:output:39%decoder/up_sampling1d/split:output:40%decoder/up_sampling1d/split:output:40%decoder/up_sampling1d/split:output:41%decoder/up_sampling1d/split:output:41%decoder/up_sampling1d/split:output:42%decoder/up_sampling1d/split:output:42%decoder/up_sampling1d/split:output:43%decoder/up_sampling1d/split:output:43%decoder/up_sampling1d/split:output:44%decoder/up_sampling1d/split:output:44%decoder/up_sampling1d/split:output:45%decoder/up_sampling1d/split:output:45%decoder/up_sampling1d/split:output:46%decoder/up_sampling1d/split:output:46*decoder/up_sampling1d/concat/axis:output:0*
N^*
T0*+
_output_shapes
:?????????^2
decoder/up_sampling1d/concat?
0decoder/conv1d_transpose/lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0decoder/conv1d_transpose/lambda_2/ExpandDims/dim?
,decoder/conv1d_transpose/lambda_2/ExpandDims
ExpandDims%decoder/up_sampling1d/concat:output:09decoder/conv1d_transpose/lambda_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2.
,decoder/conv1d_transpose/lambda_2/ExpandDims?
/decoder/conv1d_transpose/conv2d_transpose/ShapeShape5decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*
_output_shapes
:21
/decoder/conv1d_transpose/conv2d_transpose/Shape?
=decoder/conv1d_transpose/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2?
7decoder/conv1d_transpose/conv2d_transpose/strided_sliceStridedSlice8decoder/conv1d_transpose/conv2d_transpose/Shape:output:0Fdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7decoder/conv1d_transpose/conv2d_transpose/strided_slice?
1decoder/conv1d_transpose/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`23
1decoder/conv1d_transpose/conv2d_transpose/stack/1?
1decoder/conv1d_transpose/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :23
1decoder/conv1d_transpose/conv2d_transpose/stack/2?
1decoder/conv1d_transpose/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :23
1decoder/conv1d_transpose/conv2d_transpose/stack/3?
/decoder/conv1d_transpose/conv2d_transpose/stackPack@decoder/conv1d_transpose/conv2d_transpose/strided_slice:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/1:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/2:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:21
/decoder/conv1d_transpose/conv2d_transpose/stack?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack?
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1?
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2?
9decoder/conv1d_transpose/conv2d_transpose/strided_slice_1StridedSlice8decoder/conv1d_transpose/conv2d_transpose/stack:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack:output:0Jdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1:output:0Jdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9decoder/conv1d_transpose/conv2d_transpose/strided_slice_1?
Idecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpRdecoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02K
Idecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp?
:decoder/conv1d_transpose/conv2d_transpose/conv2d_transposeConv2DBackpropInput8decoder/conv1d_transpose/conv2d_transpose/stack:output:0Qdecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:05decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2<
:decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose?
@decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpIdecoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp?
1decoder/conv1d_transpose/conv2d_transpose/BiasAddBiasAddCdecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose:output:0Hdecoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`23
1decoder/conv1d_transpose/conv2d_transpose/BiasAdd?
-decoder/conv1d_transpose/conv2d_transpose/EluElu:decoder/conv1d_transpose/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`2/
-decoder/conv1d_transpose/conv2d_transpose/Elu?
)decoder/conv1d_transpose/lambda_3/SqueezeSqueeze;decoder/conv1d_transpose/conv2d_transpose/Elu:activations:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims
2+
)decoder/conv1d_transpose/lambda_3/Squeeze?
2decoder/conv1d_transpose_1/lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim?
.decoder/conv1d_transpose_1/lambda_4/ExpandDims
ExpandDims2decoder/conv1d_transpose/lambda_3/Squeeze:output:0;decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`20
.decoder/conv1d_transpose_1/lambda_4/ExpandDims?
3decoder/conv1d_transpose_1/conv2d_transpose_1/ShapeShape7decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*
_output_shapes
:25
3decoder/conv1d_transpose_1/conv2d_transpose_1/Shape?
Adecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Adecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2?
;decoder/conv1d_transpose_1/conv2d_transpose_1/strided_sliceStridedSlice<decoder/conv1d_transpose_1/conv2d_transpose_1/Shape:output:0Jdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3?
3decoder/conv1d_transpose_1/conv2d_transpose_1/stackPackDdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:25
3decoder/conv1d_transpose_1/conv2d_transpose_1/stack?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack?
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1?
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2?
=decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1StridedSlice<decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack:output:0Ndecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ndecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1?
Mdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpVdecoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02O
Mdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
>decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput<decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0Udecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:07decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2@
>decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose?
Ddecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpMdecoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02F
Ddecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
5decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAddBiasAddGdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b27
5decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd?
1decoder/conv1d_transpose_1/conv2d_transpose_1/EluElu>decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b23
1decoder/conv1d_transpose_1/conv2d_transpose_1/Elu?
+decoder/conv1d_transpose_1/lambda_5/SqueezeSqueeze?decoder/conv1d_transpose_1/conv2d_transpose_1/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2-
+decoder/conv1d_transpose_1/lambda_5/Squeeze?
decoder/lambda_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
decoder/lambda_6/ExpandDims/dim?
decoder/lambda_6/ExpandDims
ExpandDims4decoder/conv1d_transpose_1/lambda_5/Squeeze:output:0(decoder/lambda_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
decoder/lambda_6/ExpandDims?
decoder/conv_2d_transpose/ShapeShape$decoder/lambda_6/ExpandDims:output:0*
T0*
_output_shapes
:2!
decoder/conv_2d_transpose/Shape?
-decoder/conv_2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-decoder/conv_2d_transpose/strided_slice/stack?
/decoder/conv_2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/decoder/conv_2d_transpose/strided_slice/stack_1?
/decoder/conv_2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/decoder/conv_2d_transpose/strided_slice/stack_2?
'decoder/conv_2d_transpose/strided_sliceStridedSlice(decoder/conv_2d_transpose/Shape:output:06decoder/conv_2d_transpose/strided_slice/stack:output:08decoder/conv_2d_transpose/strided_slice/stack_1:output:08decoder/conv_2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'decoder/conv_2d_transpose/strided_slice?
!decoder/conv_2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d2#
!decoder/conv_2d_transpose/stack/1?
!decoder/conv_2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/conv_2d_transpose/stack/2?
!decoder/conv_2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/conv_2d_transpose/stack/3?
decoder/conv_2d_transpose/stackPack0decoder/conv_2d_transpose/strided_slice:output:0*decoder/conv_2d_transpose/stack/1:output:0*decoder/conv_2d_transpose/stack/2:output:0*decoder/conv_2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/conv_2d_transpose/stack?
/decoder/conv_2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv_2d_transpose/strided_slice_1/stack?
1decoder/conv_2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv_2d_transpose/strided_slice_1/stack_1?
1decoder/conv_2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv_2d_transpose/strided_slice_1/stack_2?
)decoder/conv_2d_transpose/strided_slice_1StridedSlice(decoder/conv_2d_transpose/stack:output:08decoder/conv_2d_transpose/strided_slice_1/stack:output:0:decoder/conv_2d_transpose/strided_slice_1/stack_1:output:0:decoder/conv_2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv_2d_transpose/strided_slice_1?
9decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpBdecoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02;
9decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp?
*decoder/conv_2d_transpose/conv2d_transposeConv2DBackpropInput(decoder/conv_2d_transpose/stack:output:0Adecoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp:value:0$decoder/lambda_6/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2,
*decoder/conv_2d_transpose/conv2d_transpose?
0decoder/conv_2d_transpose/BiasAdd/ReadVariableOpReadVariableOp9decoder_conv_2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0decoder/conv_2d_transpose/BiasAdd/ReadVariableOp?
!decoder/conv_2d_transpose/BiasAddBiasAdd3decoder/conv_2d_transpose/conv2d_transpose:output:08decoder/conv_2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2#
!decoder/conv_2d_transpose/BiasAdd?
decoder/lambda_7/SqueezeSqueeze*decoder/conv_2d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
2
decoder/lambda_7/Squeeze?
decoder/Un_Normalize/mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
decoder/Un_Normalize/mul/y?
decoder/Un_Normalize/mulMul!decoder/lambda_7/Squeeze:output:0#decoder/Un_Normalize/mul/y:output:0*
T0*+
_output_shapes
:?????????d2
decoder/Un_Normalize/mul?
decoder/Un_Normalize/add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
decoder/Un_Normalize/add/y?
decoder/Un_Normalize/addAddV2decoder/Un_Normalize/mul:z:0#decoder/Un_Normalize/add/y:output:0*
T0*+
_output_shapes
:?????????d2
decoder/Un_Normalize/addt
IdentityIdentitydecoder/Un_Normalize/add:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d:::::::::::::::::::::::::T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1
?,
?
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_4699938

inputs=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource
identity?t
lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_2/ExpandDims/dim?
lambda_2/ExpandDims
ExpandDimsinputs lambda_2/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
lambda_2/ExpandDims|
conv2d_transpose/ShapeShapelambda_2/ExpandDims:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulr
conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/add/y?
conv2d_transpose/addAddV2conv2d_transpose/mul:z:0conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/addv
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/add:z:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_2/stack?
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1?
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0lambda_2/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose/BiasAdd?
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose/Elu?
lambda_3/SqueezeSqueeze"conv2d_transpose/Elu:activations:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
lambda_3/Squeezez
IdentityIdentitylambda_3/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'???????????????????????????:::e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?-
?
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_4700058

inputs?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identity?t
lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_4/ExpandDims/dim?
lambda_4/ExpandDims
ExpandDimsinputs lambda_4/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
lambda_4/ExpandDims?
conv2d_transpose_1/ShapeShapelambda_4/ExpandDims:output:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulv
conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/add/y?
conv2d_transpose_1/addAddV2conv2d_transpose_1/mul:z:0!conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/addz
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/add:z:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_2/stack?
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1?
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0lambda_4/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose_1/Elu?
lambda_5/SqueezeSqueeze$conv2d_transpose_1/Elu:activations:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
lambda_5/Squeezez
IdentityIdentitylambda_5/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling1d_layer_call_fn_4696814

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling1d_layer_call_and_return_conditional_losses_46968082
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_layer_call_fn_4699767

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_46965302
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?I
?
D__inference_encoder_layer_call_and_return_conditional_losses_4699133

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource$
 z_matmul_readvariableop_resource%
!z_biasadd_readvariableop_resource
identity?
Std_Normalize/sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
Std_Normalize/sub/y?
Std_Normalize/subSubinputsStd_Normalize/sub/y:output:0*
T0*+
_output_shapes
:?????????d2
Std_Normalize/sub?
Std_Normalize/truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
Std_Normalize/truediv/y?
Std_Normalize/truedivRealDivStd_Normalize/sub:z:0 Std_Normalize/truediv/y:output:0*
T0*+
_output_shapes
:?????????d2
Std_Normalize/truedivp
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/ExpandDims/dim?
lambda/ExpandDims
ExpandDimsStd_Normalize/truediv:z:0lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2
lambda/ExpandDims?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dlambda/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2
conv2d/BiasAddr

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2

conv2d/Elu?
lambda_1/SqueezeSqueezeconv2d/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2
lambda_1/Squeeze?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimslambda_1/Squeeze:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
conv1d/BiasAddn

conv1d/EluEluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2

conv1d/Elu?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Elu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2
conv1d_1/BiasAddt
conv1d_1/EluEluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^2
conv1d_1/Elu?
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dim?
average_pooling1d/ExpandDims
ExpandDimsconv1d_1/Elu:activations:0)average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2
average_pooling1d/ExpandDims?
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2
average_pooling1d/AvgPool?
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2
average_pooling1d/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
flatten/Const?
flatten/ReshapeReshape"average_pooling1d/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
dense/BiasAddg
	dense/EluEludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
	dense/Elu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Elu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddm
dense_1/EluEludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Elu?
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z/MatMul/ReadVariableOp?
z/MatMulMatMuldense_1/Elu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

z/MatMul?
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z/BiasAdd/ReadVariableOp?
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	z/BiasAddf
IdentityIdentityz/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d:::::::::::::S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
F
*__inference_lambda_6_layer_call_fn_4700098

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_46972652
PartitionedCall}
IdentityIdentityPartitionedCall:output:0*
T0*8
_output_shapes&
$:"??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?-
?
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_4697228

inputs?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identity?t
lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_4/ExpandDims/dim?
lambda_4/ExpandDims
ExpandDimsinputs lambda_4/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
lambda_4/ExpandDims?
conv2d_transpose_1/ShapeShapelambda_4/ExpandDims:output:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulv
conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/add/y?
conv2d_transpose_1/addAddV2conv2d_transpose_1/mul:z:0!conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/addz
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/add:z:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_2/stack?
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1?
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0lambda_4/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose_1/Elu?
lambda_5/SqueezeSqueeze$conv2d_transpose_1/Elu:activations:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
lambda_5/Squeezez
IdentityIdentitylambda_5/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
D__inference_dense_1_layer_call_and_return_conditional_losses_4699798

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????f:::O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?
~
)__inference_dense_4_layer_call_fn_4699886

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46970292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????f::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?
F
*__inference_lambda_7_layer_call_fn_4700118

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_46972932
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?,
?
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_4697101

inputs=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource
identity?t
lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_2/ExpandDims/dim?
lambda_2/ExpandDims
ExpandDimsinputs lambda_2/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
lambda_2/ExpandDims|
conv2d_transpose/ShapeShapelambda_2/ExpandDims:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulr
conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/add/y?
conv2d_transpose/addAddV2conv2d_transpose/mul:z:0conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/addv
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/add:z:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_2/stack?
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1?
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0lambda_2/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose/BiasAdd?
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose/Elu?
lambda_3/SqueezeSqueeze"conv2d_transpose/Elu:activations:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
lambda_3/Squeezez
IdentityIdentitylambda_3/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'???????????????????????????:::e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
)__inference_encoder_layer_call_fn_4696795
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_46967682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????d
'
_user_specified_nameencoder_input
?	
?
)__inference_decoder_layer_call_fn_4697431
z
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46974042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namez
?
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_4697265

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2

ExpandDimsx
IdentityIdentityExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
F
*__inference_lambda_1_layer_call_fn_4699706

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_46964462
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????b:W S
/
_output_shapes
:?????????b
 
_user_specified_nameinputs
?
F
*__inference_lambda_7_layer_call_fn_4700113

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_46972882
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
>__inference_z_layer_call_and_return_conditional_losses_4696602

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv1d_transpose_layer_call_fn_4699990

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_46971352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?-
?
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_4700024

inputs?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identity?t
lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_4/ExpandDims/dim?
lambda_4/ExpandDims
ExpandDimsinputs lambda_4/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
lambda_4/ExpandDims?
conv2d_transpose_1/ShapeShapelambda_4/ExpandDims:output:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulv
conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/add/y?
conv2d_transpose_1/addAddV2conv2d_transpose_1/mul:z:0!conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/addz
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/add:z:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_2/stack?
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1?
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0lambda_4/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose_1/Elu?
lambda_5/SqueezeSqueeze$conv2d_transpose_1/Elu:activations:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
lambda_5/Squeezez
IdentityIdentitylambda_5/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_4696349
input_1F
Bparticle_autoencoder_encoder_conv2d_conv2d_readvariableop_resourceG
Cparticle_autoencoder_encoder_conv2d_biasadd_readvariableop_resourceS
Oparticle_autoencoder_encoder_conv1d_conv1d_expanddims_1_readvariableop_resourceG
Cparticle_autoencoder_encoder_conv1d_biasadd_readvariableop_resourceU
Qparticle_autoencoder_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resourceI
Eparticle_autoencoder_encoder_conv1d_1_biasadd_readvariableop_resourceE
Aparticle_autoencoder_encoder_dense_matmul_readvariableop_resourceF
Bparticle_autoencoder_encoder_dense_biasadd_readvariableop_resourceG
Cparticle_autoencoder_encoder_dense_1_matmul_readvariableop_resourceH
Dparticle_autoencoder_encoder_dense_1_biasadd_readvariableop_resourceA
=particle_autoencoder_encoder_z_matmul_readvariableop_resourceB
>particle_autoencoder_encoder_z_biasadd_readvariableop_resourceG
Cparticle_autoencoder_decoder_dense_2_matmul_readvariableop_resourceH
Dparticle_autoencoder_decoder_dense_2_biasadd_readvariableop_resourceG
Cparticle_autoencoder_decoder_dense_3_matmul_readvariableop_resourceH
Dparticle_autoencoder_decoder_dense_3_biasadd_readvariableop_resourceG
Cparticle_autoencoder_decoder_dense_4_matmul_readvariableop_resourceH
Dparticle_autoencoder_decoder_dense_4_biasadd_readvariableop_resourcek
gparticle_autoencoder_decoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resourceb
^particle_autoencoder_decoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resourceo
kparticle_autoencoder_decoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourcef
bparticle_autoencoder_decoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource[
Wparticle_autoencoder_decoder_conv_2d_transpose_conv2d_transpose_readvariableop_resourceR
Nparticle_autoencoder_decoder_conv_2d_transpose_biasadd_readvariableop_resource
identity??
0particle_autoencoder/encoder/Std_Normalize/sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@22
0particle_autoencoder/encoder/Std_Normalize/sub/y?
.particle_autoencoder/encoder/Std_Normalize/subSubinput_19particle_autoencoder/encoder/Std_Normalize/sub/y:output:0*
T0*+
_output_shapes
:?????????d20
.particle_autoencoder/encoder/Std_Normalize/sub?
4particle_autoencoder/encoder/Std_Normalize/truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A26
4particle_autoencoder/encoder/Std_Normalize/truediv/y?
2particle_autoencoder/encoder/Std_Normalize/truedivRealDiv2particle_autoencoder/encoder/Std_Normalize/sub:z:0=particle_autoencoder/encoder/Std_Normalize/truediv/y:output:0*
T0*+
_output_shapes
:?????????d24
2particle_autoencoder/encoder/Std_Normalize/truediv?
2particle_autoencoder/encoder/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2particle_autoencoder/encoder/lambda/ExpandDims/dim?
.particle_autoencoder/encoder/lambda/ExpandDims
ExpandDims6particle_autoencoder/encoder/Std_Normalize/truediv:z:0;particle_autoencoder/encoder/lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d20
.particle_autoencoder/encoder/lambda/ExpandDims?
9particle_autoencoder/encoder/conv2d/Conv2D/ReadVariableOpReadVariableOpBparticle_autoencoder_encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02;
9particle_autoencoder/encoder/conv2d/Conv2D/ReadVariableOp?
*particle_autoencoder/encoder/conv2d/Conv2DConv2D7particle_autoencoder/encoder/lambda/ExpandDims:output:0Aparticle_autoencoder/encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2,
*particle_autoencoder/encoder/conv2d/Conv2D?
:particle_autoencoder/encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOpCparticle_autoencoder_encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:particle_autoencoder/encoder/conv2d/BiasAdd/ReadVariableOp?
+particle_autoencoder/encoder/conv2d/BiasAddBiasAdd3particle_autoencoder/encoder/conv2d/Conv2D:output:0Bparticle_autoencoder/encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2-
+particle_autoencoder/encoder/conv2d/BiasAdd?
'particle_autoencoder/encoder/conv2d/EluElu4particle_autoencoder/encoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2)
'particle_autoencoder/encoder/conv2d/Elu?
-particle_autoencoder/encoder/lambda_1/SqueezeSqueeze5particle_autoencoder/encoder/conv2d/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2/
-particle_autoencoder/encoder/lambda_1/Squeeze?
9particle_autoencoder/encoder/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2;
9particle_autoencoder/encoder/conv1d/conv1d/ExpandDims/dim?
5particle_autoencoder/encoder/conv1d/conv1d/ExpandDims
ExpandDims6particle_autoencoder/encoder/lambda_1/Squeeze:output:0Bparticle_autoencoder/encoder/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b27
5particle_autoencoder/encoder/conv1d/conv1d/ExpandDims?
Fparticle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpOparticle_autoencoder_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02H
Fparticle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
;particle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2=
;particle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1/dim?
7particle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1
ExpandDimsNparticle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0Dparticle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:29
7particle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1?
*particle_autoencoder/encoder/conv1d/conv1dConv2D>particle_autoencoder/encoder/conv1d/conv1d/ExpandDims:output:0@particle_autoencoder/encoder/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2,
*particle_autoencoder/encoder/conv1d/conv1d?
2particle_autoencoder/encoder/conv1d/conv1d/SqueezeSqueeze3particle_autoencoder/encoder/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????24
2particle_autoencoder/encoder/conv1d/conv1d/Squeeze?
:particle_autoencoder/encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOpCparticle_autoencoder_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:particle_autoencoder/encoder/conv1d/BiasAdd/ReadVariableOp?
+particle_autoencoder/encoder/conv1d/BiasAddBiasAdd;particle_autoencoder/encoder/conv1d/conv1d/Squeeze:output:0Bparticle_autoencoder/encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2-
+particle_autoencoder/encoder/conv1d/BiasAdd?
'particle_autoencoder/encoder/conv1d/EluElu4particle_autoencoder/encoder/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2)
'particle_autoencoder/encoder/conv1d/Elu?
;particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2=
;particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims/dim?
7particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims
ExpandDims5particle_autoencoder/encoder/conv1d/Elu:activations:0Dparticle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`29
7particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims?
Hparticle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpQparticle_autoencoder_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02J
Hparticle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
=particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2?
=particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1/dim?
9particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1
ExpandDimsPparticle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Fparticle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2;
9particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1?
,particle_autoencoder/encoder/conv1d_1/conv1dConv2D@particle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims:output:0Bparticle_autoencoder/encoder/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2.
,particle_autoencoder/encoder/conv1d_1/conv1d?
4particle_autoencoder/encoder/conv1d_1/conv1d/SqueezeSqueeze5particle_autoencoder/encoder/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????26
4particle_autoencoder/encoder/conv1d_1/conv1d/Squeeze?
<particle_autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOpEparticle_autoencoder_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<particle_autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOp?
-particle_autoencoder/encoder/conv1d_1/BiasAddBiasAdd=particle_autoencoder/encoder/conv1d_1/conv1d/Squeeze:output:0Dparticle_autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2/
-particle_autoencoder/encoder/conv1d_1/BiasAdd?
)particle_autoencoder/encoder/conv1d_1/EluElu6particle_autoencoder/encoder/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^2+
)particle_autoencoder/encoder/conv1d_1/Elu?
=particle_autoencoder/encoder/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2?
=particle_autoencoder/encoder/average_pooling1d/ExpandDims/dim?
9particle_autoencoder/encoder/average_pooling1d/ExpandDims
ExpandDims7particle_autoencoder/encoder/conv1d_1/Elu:activations:0Fparticle_autoencoder/encoder/average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2;
9particle_autoencoder/encoder/average_pooling1d/ExpandDims?
6particle_autoencoder/encoder/average_pooling1d/AvgPoolAvgPoolBparticle_autoencoder/encoder/average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
28
6particle_autoencoder/encoder/average_pooling1d/AvgPool?
6particle_autoencoder/encoder/average_pooling1d/SqueezeSqueeze?particle_autoencoder/encoder/average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
28
6particle_autoencoder/encoder/average_pooling1d/Squeeze?
*particle_autoencoder/encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2,
*particle_autoencoder/encoder/flatten/Const?
,particle_autoencoder/encoder/flatten/ReshapeReshape?particle_autoencoder/encoder/average_pooling1d/Squeeze:output:03particle_autoencoder/encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2.
,particle_autoencoder/encoder/flatten/Reshape?
8particle_autoencoder/encoder/dense/MatMul/ReadVariableOpReadVariableOpAparticle_autoencoder_encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02:
8particle_autoencoder/encoder/dense/MatMul/ReadVariableOp?
)particle_autoencoder/encoder/dense/MatMulMatMul5particle_autoencoder/encoder/flatten/Reshape:output:0@particle_autoencoder/encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2+
)particle_autoencoder/encoder/dense/MatMul?
9particle_autoencoder/encoder/dense/BiasAdd/ReadVariableOpReadVariableOpBparticle_autoencoder_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02;
9particle_autoencoder/encoder/dense/BiasAdd/ReadVariableOp?
*particle_autoencoder/encoder/dense/BiasAddBiasAdd3particle_autoencoder/encoder/dense/MatMul:product:0Aparticle_autoencoder/encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2,
*particle_autoencoder/encoder/dense/BiasAdd?
&particle_autoencoder/encoder/dense/EluElu3particle_autoencoder/encoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2(
&particle_autoencoder/encoder/dense/Elu?
:particle_autoencoder/encoder/dense_1/MatMul/ReadVariableOpReadVariableOpCparticle_autoencoder_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02<
:particle_autoencoder/encoder/dense_1/MatMul/ReadVariableOp?
+particle_autoencoder/encoder/dense_1/MatMulMatMul4particle_autoencoder/encoder/dense/Elu:activations:0Bparticle_autoencoder/encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+particle_autoencoder/encoder/dense_1/MatMul?
;particle_autoencoder/encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpDparticle_autoencoder_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;particle_autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp?
,particle_autoencoder/encoder/dense_1/BiasAddBiasAdd5particle_autoencoder/encoder/dense_1/MatMul:product:0Cparticle_autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,particle_autoencoder/encoder/dense_1/BiasAdd?
(particle_autoencoder/encoder/dense_1/EluElu5particle_autoencoder/encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(particle_autoencoder/encoder/dense_1/Elu?
4particle_autoencoder/encoder/z/MatMul/ReadVariableOpReadVariableOp=particle_autoencoder_encoder_z_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4particle_autoencoder/encoder/z/MatMul/ReadVariableOp?
%particle_autoencoder/encoder/z/MatMulMatMul6particle_autoencoder/encoder/dense_1/Elu:activations:0<particle_autoencoder/encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%particle_autoencoder/encoder/z/MatMul?
5particle_autoencoder/encoder/z/BiasAdd/ReadVariableOpReadVariableOp>particle_autoencoder_encoder_z_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5particle_autoencoder/encoder/z/BiasAdd/ReadVariableOp?
&particle_autoencoder/encoder/z/BiasAddBiasAdd/particle_autoencoder/encoder/z/MatMul:product:0=particle_autoencoder/encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&particle_autoencoder/encoder/z/BiasAdd?
:particle_autoencoder/decoder/dense_2/MatMul/ReadVariableOpReadVariableOpCparticle_autoencoder_decoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:particle_autoencoder/decoder/dense_2/MatMul/ReadVariableOp?
+particle_autoencoder/decoder/dense_2/MatMulMatMul/particle_autoencoder/encoder/z/BiasAdd:output:0Bparticle_autoencoder/decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+particle_autoencoder/decoder/dense_2/MatMul?
;particle_autoencoder/decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOpDparticle_autoencoder_decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;particle_autoencoder/decoder/dense_2/BiasAdd/ReadVariableOp?
,particle_autoencoder/decoder/dense_2/BiasAddBiasAdd5particle_autoencoder/decoder/dense_2/MatMul:product:0Cparticle_autoencoder/decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,particle_autoencoder/decoder/dense_2/BiasAdd?
(particle_autoencoder/decoder/dense_2/EluElu5particle_autoencoder/decoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2*
(particle_autoencoder/decoder/dense_2/Elu?
:particle_autoencoder/decoder/dense_3/MatMul/ReadVariableOpReadVariableOpCparticle_autoencoder_decoder_dense_3_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02<
:particle_autoencoder/decoder/dense_3/MatMul/ReadVariableOp?
+particle_autoencoder/decoder/dense_3/MatMulMatMul6particle_autoencoder/decoder/dense_2/Elu:activations:0Bparticle_autoencoder/decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2-
+particle_autoencoder/decoder/dense_3/MatMul?
;particle_autoencoder/decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOpDparticle_autoencoder_decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02=
;particle_autoencoder/decoder/dense_3/BiasAdd/ReadVariableOp?
,particle_autoencoder/decoder/dense_3/BiasAddBiasAdd5particle_autoencoder/decoder/dense_3/MatMul:product:0Cparticle_autoencoder/decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2.
,particle_autoencoder/decoder/dense_3/BiasAdd?
(particle_autoencoder/decoder/dense_3/EluElu5particle_autoencoder/decoder/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2*
(particle_autoencoder/decoder/dense_3/Elu?
:particle_autoencoder/decoder/dense_4/MatMul/ReadVariableOpReadVariableOpCparticle_autoencoder_decoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02<
:particle_autoencoder/decoder/dense_4/MatMul/ReadVariableOp?
+particle_autoencoder/decoder/dense_4/MatMulMatMul6particle_autoencoder/decoder/dense_3/Elu:activations:0Bparticle_autoencoder/decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+particle_autoencoder/decoder/dense_4/MatMul?
;particle_autoencoder/decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOpDparticle_autoencoder_decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;particle_autoencoder/decoder/dense_4/BiasAdd/ReadVariableOp?
,particle_autoencoder/decoder/dense_4/BiasAddBiasAdd5particle_autoencoder/decoder/dense_4/MatMul:product:0Cparticle_autoencoder/decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,particle_autoencoder/decoder/dense_4/BiasAdd?
(particle_autoencoder/decoder/dense_4/EluElu5particle_autoencoder/decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2*
(particle_autoencoder/decoder/dense_4/Elu?
*particle_autoencoder/decoder/reshape/ShapeShape6particle_autoencoder/decoder/dense_4/Elu:activations:0*
T0*
_output_shapes
:2,
*particle_autoencoder/decoder/reshape/Shape?
8particle_autoencoder/decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8particle_autoencoder/decoder/reshape/strided_slice/stack?
:particle_autoencoder/decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:particle_autoencoder/decoder/reshape/strided_slice/stack_1?
:particle_autoencoder/decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:particle_autoencoder/decoder/reshape/strided_slice/stack_2?
2particle_autoencoder/decoder/reshape/strided_sliceStridedSlice3particle_autoencoder/decoder/reshape/Shape:output:0Aparticle_autoencoder/decoder/reshape/strided_slice/stack:output:0Cparticle_autoencoder/decoder/reshape/strided_slice/stack_1:output:0Cparticle_autoencoder/decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2particle_autoencoder/decoder/reshape/strided_slice?
4particle_autoencoder/decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/26
4particle_autoencoder/decoder/reshape/Reshape/shape/1?
4particle_autoencoder/decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :26
4particle_autoencoder/decoder/reshape/Reshape/shape/2?
2particle_autoencoder/decoder/reshape/Reshape/shapePack;particle_autoencoder/decoder/reshape/strided_slice:output:0=particle_autoencoder/decoder/reshape/Reshape/shape/1:output:0=particle_autoencoder/decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:24
2particle_autoencoder/decoder/reshape/Reshape/shape?
,particle_autoencoder/decoder/reshape/ReshapeReshape6particle_autoencoder/decoder/dense_4/Elu:activations:0;particle_autoencoder/decoder/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????/2.
,particle_autoencoder/decoder/reshape/Reshape?
0particle_autoencoder/decoder/up_sampling1d/ConstConst*
_output_shapes
: *
dtype0*
value	B :/22
0particle_autoencoder/decoder/up_sampling1d/Const?
:particle_autoencoder/decoder/up_sampling1d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:particle_autoencoder/decoder/up_sampling1d/split/split_dim?

0particle_autoencoder/decoder/up_sampling1d/splitSplitCparticle_autoencoder/decoder/up_sampling1d/split/split_dim:output:05particle_autoencoder/decoder/reshape/Reshape:output:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split/22
0particle_autoencoder/decoder/up_sampling1d/split?
6particle_autoencoder/decoder/up_sampling1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :28
6particle_autoencoder/decoder/up_sampling1d/concat/axis?-
1particle_autoencoder/decoder/up_sampling1d/concatConcatV29particle_autoencoder/decoder/up_sampling1d/split:output:09particle_autoencoder/decoder/up_sampling1d/split:output:09particle_autoencoder/decoder/up_sampling1d/split:output:19particle_autoencoder/decoder/up_sampling1d/split:output:19particle_autoencoder/decoder/up_sampling1d/split:output:29particle_autoencoder/decoder/up_sampling1d/split:output:29particle_autoencoder/decoder/up_sampling1d/split:output:39particle_autoencoder/decoder/up_sampling1d/split:output:39particle_autoencoder/decoder/up_sampling1d/split:output:49particle_autoencoder/decoder/up_sampling1d/split:output:49particle_autoencoder/decoder/up_sampling1d/split:output:59particle_autoencoder/decoder/up_sampling1d/split:output:59particle_autoencoder/decoder/up_sampling1d/split:output:69particle_autoencoder/decoder/up_sampling1d/split:output:69particle_autoencoder/decoder/up_sampling1d/split:output:79particle_autoencoder/decoder/up_sampling1d/split:output:79particle_autoencoder/decoder/up_sampling1d/split:output:89particle_autoencoder/decoder/up_sampling1d/split:output:89particle_autoencoder/decoder/up_sampling1d/split:output:99particle_autoencoder/decoder/up_sampling1d/split:output:9:particle_autoencoder/decoder/up_sampling1d/split:output:10:particle_autoencoder/decoder/up_sampling1d/split:output:10:particle_autoencoder/decoder/up_sampling1d/split:output:11:particle_autoencoder/decoder/up_sampling1d/split:output:11:particle_autoencoder/decoder/up_sampling1d/split:output:12:particle_autoencoder/decoder/up_sampling1d/split:output:12:particle_autoencoder/decoder/up_sampling1d/split:output:13:particle_autoencoder/decoder/up_sampling1d/split:output:13:particle_autoencoder/decoder/up_sampling1d/split:output:14:particle_autoencoder/decoder/up_sampling1d/split:output:14:particle_autoencoder/decoder/up_sampling1d/split:output:15:particle_autoencoder/decoder/up_sampling1d/split:output:15:particle_autoencoder/decoder/up_sampling1d/split:output:16:particle_autoencoder/decoder/up_sampling1d/split:output:16:particle_autoencoder/decoder/up_sampling1d/split:output:17:particle_autoencoder/decoder/up_sampling1d/split:output:17:particle_autoencoder/decoder/up_sampling1d/split:output:18:particle_autoencoder/decoder/up_sampling1d/split:output:18:particle_autoencoder/decoder/up_sampling1d/split:output:19:particle_autoencoder/decoder/up_sampling1d/split:output:19:particle_autoencoder/decoder/up_sampling1d/split:output:20:particle_autoencoder/decoder/up_sampling1d/split:output:20:particle_autoencoder/decoder/up_sampling1d/split:output:21:particle_autoencoder/decoder/up_sampling1d/split:output:21:particle_autoencoder/decoder/up_sampling1d/split:output:22:particle_autoencoder/decoder/up_sampling1d/split:output:22:particle_autoencoder/decoder/up_sampling1d/split:output:23:particle_autoencoder/decoder/up_sampling1d/split:output:23:particle_autoencoder/decoder/up_sampling1d/split:output:24:particle_autoencoder/decoder/up_sampling1d/split:output:24:particle_autoencoder/decoder/up_sampling1d/split:output:25:particle_autoencoder/decoder/up_sampling1d/split:output:25:particle_autoencoder/decoder/up_sampling1d/split:output:26:particle_autoencoder/decoder/up_sampling1d/split:output:26:particle_autoencoder/decoder/up_sampling1d/split:output:27:particle_autoencoder/decoder/up_sampling1d/split:output:27:particle_autoencoder/decoder/up_sampling1d/split:output:28:particle_autoencoder/decoder/up_sampling1d/split:output:28:particle_autoencoder/decoder/up_sampling1d/split:output:29:particle_autoencoder/decoder/up_sampling1d/split:output:29:particle_autoencoder/decoder/up_sampling1d/split:output:30:particle_autoencoder/decoder/up_sampling1d/split:output:30:particle_autoencoder/decoder/up_sampling1d/split:output:31:particle_autoencoder/decoder/up_sampling1d/split:output:31:particle_autoencoder/decoder/up_sampling1d/split:output:32:particle_autoencoder/decoder/up_sampling1d/split:output:32:particle_autoencoder/decoder/up_sampling1d/split:output:33:particle_autoencoder/decoder/up_sampling1d/split:output:33:particle_autoencoder/decoder/up_sampling1d/split:output:34:particle_autoencoder/decoder/up_sampling1d/split:output:34:particle_autoencoder/decoder/up_sampling1d/split:output:35:particle_autoencoder/decoder/up_sampling1d/split:output:35:particle_autoencoder/decoder/up_sampling1d/split:output:36:particle_autoencoder/decoder/up_sampling1d/split:output:36:particle_autoencoder/decoder/up_sampling1d/split:output:37:particle_autoencoder/decoder/up_sampling1d/split:output:37:particle_autoencoder/decoder/up_sampling1d/split:output:38:particle_autoencoder/decoder/up_sampling1d/split:output:38:particle_autoencoder/decoder/up_sampling1d/split:output:39:particle_autoencoder/decoder/up_sampling1d/split:output:39:particle_autoencoder/decoder/up_sampling1d/split:output:40:particle_autoencoder/decoder/up_sampling1d/split:output:40:particle_autoencoder/decoder/up_sampling1d/split:output:41:particle_autoencoder/decoder/up_sampling1d/split:output:41:particle_autoencoder/decoder/up_sampling1d/split:output:42:particle_autoencoder/decoder/up_sampling1d/split:output:42:particle_autoencoder/decoder/up_sampling1d/split:output:43:particle_autoencoder/decoder/up_sampling1d/split:output:43:particle_autoencoder/decoder/up_sampling1d/split:output:44:particle_autoencoder/decoder/up_sampling1d/split:output:44:particle_autoencoder/decoder/up_sampling1d/split:output:45:particle_autoencoder/decoder/up_sampling1d/split:output:45:particle_autoencoder/decoder/up_sampling1d/split:output:46:particle_autoencoder/decoder/up_sampling1d/split:output:46?particle_autoencoder/decoder/up_sampling1d/concat/axis:output:0*
N^*
T0*+
_output_shapes
:?????????^23
1particle_autoencoder/decoder/up_sampling1d/concat?
Eparticle_autoencoder/decoder/conv1d_transpose/lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2G
Eparticle_autoencoder/decoder/conv1d_transpose/lambda_2/ExpandDims/dim?
Aparticle_autoencoder/decoder/conv1d_transpose/lambda_2/ExpandDims
ExpandDims:particle_autoencoder/decoder/up_sampling1d/concat:output:0Nparticle_autoencoder/decoder/conv1d_transpose/lambda_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2C
Aparticle_autoencoder/decoder/conv1d_transpose/lambda_2/ExpandDims?
Dparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/ShapeShapeJparticle_autoencoder/decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*
_output_shapes
:2F
Dparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/Shape?
Rparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2T
Rparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack?
Tparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2V
Tparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1?
Tparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2V
Tparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2?
Lparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_sliceStridedSliceMparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/Shape:output:0[particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack:output:0]particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1:output:0]particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2N
Lparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice?
Fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2H
Fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/1?
Fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2H
Fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/2?
Fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2H
Fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/3?
Dparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stackPackUparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice:output:0Oparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/1:output:0Oparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/2:output:0Oparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2F
Dparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack?
Tparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
Tparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack?
Vparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2X
Vparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1?
Vparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
Vparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2?
Nparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1StridedSliceMparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack:output:0]particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack:output:0_particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1:output:0_particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2P
Nparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/strided_slice_1?
^particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpgparticle_autoencoder_decoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02`
^particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp?
Oparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/conv2d_transposeConv2DBackpropInputMparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/stack:output:0fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Jparticle_autoencoder/decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2Q
Oparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose?
Uparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp^particle_autoencoder_decoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
Uparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp?
Fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/BiasAddBiasAddXparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose:output:0]particle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`2H
Fparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/BiasAdd?
Bparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/EluEluOparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`2D
Bparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/Elu?
>particle_autoencoder/decoder/conv1d_transpose/lambda_3/SqueezeSqueezePparticle_autoencoder/decoder/conv1d_transpose/conv2d_transpose/Elu:activations:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims
2@
>particle_autoencoder/decoder/conv1d_transpose/lambda_3/Squeeze?
Gparticle_autoencoder/decoder/conv1d_transpose_1/lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2I
Gparticle_autoencoder/decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim?
Cparticle_autoencoder/decoder/conv1d_transpose_1/lambda_4/ExpandDims
ExpandDimsGparticle_autoencoder/decoder/conv1d_transpose/lambda_3/Squeeze:output:0Pparticle_autoencoder/decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2E
Cparticle_autoencoder/decoder/conv1d_transpose_1/lambda_4/ExpandDims?
Hparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/ShapeShapeLparticle_autoencoder/decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*
_output_shapes
:2J
Hparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/Shape?
Vparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2X
Vparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack?
Xparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1?
Xparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2?
Pparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_sliceStridedSliceQparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/Shape:output:0_particle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack:output:0aparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1:output:0aparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2R
Pparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice?
Jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b2L
Jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1?
Jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2L
Jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2?
Jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2L
Jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3?
Hparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stackPackYparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice:output:0Sparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1:output:0Sparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2:output:0Sparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2J
Hparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack?
Xparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Z
Xparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack?
Zparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2\
Zparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1?
Zparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2\
Zparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2?
Rparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1StridedSliceQparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0aparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack:output:0cparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0cparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2T
Rparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1?
bparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpkparticle_autoencoder_decoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02d
bparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
Sparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInputQparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Lparticle_autoencoder/decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2U
Sparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose?
Yparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpbparticle_autoencoder_decoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02[
Yparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
Jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAddBiasAdd\particle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose:output:0aparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2L
Jparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd?
Fparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/EluEluSparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2H
Fparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/Elu?
@particle_autoencoder/decoder/conv1d_transpose_1/lambda_5/SqueezeSqueezeTparticle_autoencoder/decoder/conv1d_transpose_1/conv2d_transpose_1/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2B
@particle_autoencoder/decoder/conv1d_transpose_1/lambda_5/Squeeze?
4particle_autoencoder/decoder/lambda_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :26
4particle_autoencoder/decoder/lambda_6/ExpandDims/dim?
0particle_autoencoder/decoder/lambda_6/ExpandDims
ExpandDimsIparticle_autoencoder/decoder/conv1d_transpose_1/lambda_5/Squeeze:output:0=particle_autoencoder/decoder/lambda_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b22
0particle_autoencoder/decoder/lambda_6/ExpandDims?
4particle_autoencoder/decoder/conv_2d_transpose/ShapeShape9particle_autoencoder/decoder/lambda_6/ExpandDims:output:0*
T0*
_output_shapes
:26
4particle_autoencoder/decoder/conv_2d_transpose/Shape?
Bparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stack?
Dparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stack_1?
Dparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stack_2?
<particle_autoencoder/decoder/conv_2d_transpose/strided_sliceStridedSlice=particle_autoencoder/decoder/conv_2d_transpose/Shape:output:0Kparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stack:output:0Mparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stack_1:output:0Mparticle_autoencoder/decoder/conv_2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<particle_autoencoder/decoder/conv_2d_transpose/strided_slice?
6particle_autoencoder/decoder/conv_2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d28
6particle_autoencoder/decoder/conv_2d_transpose/stack/1?
6particle_autoencoder/decoder/conv_2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :28
6particle_autoencoder/decoder/conv_2d_transpose/stack/2?
6particle_autoencoder/decoder/conv_2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :28
6particle_autoencoder/decoder/conv_2d_transpose/stack/3?
4particle_autoencoder/decoder/conv_2d_transpose/stackPackEparticle_autoencoder/decoder/conv_2d_transpose/strided_slice:output:0?particle_autoencoder/decoder/conv_2d_transpose/stack/1:output:0?particle_autoencoder/decoder/conv_2d_transpose/stack/2:output:0?particle_autoencoder/decoder/conv_2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:26
4particle_autoencoder/decoder/conv_2d_transpose/stack?
Dparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stack?
Fparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stack_1?
Fparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stack_2?
>particle_autoencoder/decoder/conv_2d_transpose/strided_slice_1StridedSlice=particle_autoencoder/decoder/conv_2d_transpose/stack:output:0Mparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stack:output:0Oparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stack_1:output:0Oparticle_autoencoder/decoder/conv_2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>particle_autoencoder/decoder/conv_2d_transpose/strided_slice_1?
Nparticle_autoencoder/decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpWparticle_autoencoder_decoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nparticle_autoencoder/decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp?
?particle_autoencoder/decoder/conv_2d_transpose/conv2d_transposeConv2DBackpropInput=particle_autoencoder/decoder/conv_2d_transpose/stack:output:0Vparticle_autoencoder/decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp:value:09particle_autoencoder/decoder/lambda_6/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2A
?particle_autoencoder/decoder/conv_2d_transpose/conv2d_transpose?
Eparticle_autoencoder/decoder/conv_2d_transpose/BiasAdd/ReadVariableOpReadVariableOpNparticle_autoencoder_decoder_conv_2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Eparticle_autoencoder/decoder/conv_2d_transpose/BiasAdd/ReadVariableOp?
6particle_autoencoder/decoder/conv_2d_transpose/BiasAddBiasAddHparticle_autoencoder/decoder/conv_2d_transpose/conv2d_transpose:output:0Mparticle_autoencoder/decoder/conv_2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d28
6particle_autoencoder/decoder/conv_2d_transpose/BiasAdd?
-particle_autoencoder/decoder/lambda_7/SqueezeSqueeze?particle_autoencoder/decoder/conv_2d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
2/
-particle_autoencoder/decoder/lambda_7/Squeeze?
/particle_autoencoder/decoder/Un_Normalize/mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A21
/particle_autoencoder/decoder/Un_Normalize/mul/y?
-particle_autoencoder/decoder/Un_Normalize/mulMul6particle_autoencoder/decoder/lambda_7/Squeeze:output:08particle_autoencoder/decoder/Un_Normalize/mul/y:output:0*
T0*+
_output_shapes
:?????????d2/
-particle_autoencoder/decoder/Un_Normalize/mul?
/particle_autoencoder/decoder/Un_Normalize/add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@21
/particle_autoencoder/decoder/Un_Normalize/add/y?
-particle_autoencoder/decoder/Un_Normalize/addAddV21particle_autoencoder/decoder/Un_Normalize/mul:z:08particle_autoencoder/decoder/Un_Normalize/add/y:output:0*
T0*+
_output_shapes
:?????????d2/
-particle_autoencoder/decoder/Un_Normalize/add?
IdentityIdentity1particle_autoencoder/decoder/Un_Normalize/add:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d:::::::::::::::::::::::::T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1
?
}
(__inference_conv1d_layer_call_fn_4699731

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_46964752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????b::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????b
 
_user_specified_nameinputs
?.
?
D__inference_encoder_layer_call_and_return_conditional_losses_4696658
encoder_input
conv2d_4696624
conv2d_4696626
conv1d_4696630
conv1d_4696632
conv1d_1_4696635
conv1d_1_4696637
dense_4696642
dense_4696644
dense_1_4696647
dense_1_4696649
	z_4696652
	z_4696654
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?z/StatefulPartitionedCall?
Std_Normalize/PartitionedCallPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_46963762
Std_Normalize/PartitionedCall?
lambda/PartitionedCallPartitionedCall&Std_Normalize/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_46963962
lambda/PartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_4696624conv2d_4696626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_46964202 
conv2d/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_46964462
lambda_1/PartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_4696630conv1d_4696632*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_46964752 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_4696635conv1d_1_4696637*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_46965072"
 conv1d_1/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_layer_call_and_return_conditional_losses_46963582#
!average_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall*average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_46965302
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4696642dense_4696644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_46965492
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4696647dense_1_4696649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_46965762!
dense_1/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0	z_4696652	z_4696654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_z_layer_call_and_return_conditional_losses_46966022
z/StatefulPartitionedCall?
IdentityIdentity"z/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????d
'
_user_specified_nameencoder_input
?2
?
D__inference_decoder_layer_call_and_return_conditional_losses_4697404

inputs
dense_2_4697368
dense_2_4697370
dense_3_4697373
dense_3_4697375
dense_4_4697378
dense_4_4697380
conv1d_transpose_4697385
conv1d_transpose_4697387
conv1d_transpose_1_4697390
conv1d_transpose_1_4697392
conv_2d_transpose_4697396
conv_2d_transpose_4697398
identity??(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?)conv_2d_transpose/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_4697368dense_2_4697370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46969752!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4697373dense_3_4697375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46970022!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_4697378dense_4_4697380*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46970292!
dense_4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46970582
reshape/PartitionedCall?
up_sampling1d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling1d_layer_call_and_return_conditional_losses_46968082
up_sampling1d/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall&up_sampling1d/PartitionedCall:output:0conv1d_transpose_4697385conv1d_transpose_4697387*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_46971012*
(conv1d_transpose/StatefulPartitionedCall?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_4697390conv1d_transpose_1_4697392*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_46971942,
*conv1d_transpose_1/StatefulPartitionedCall?
lambda_6/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_46972592
lambda_6/PartitionedCall?
)conv_2d_transpose/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0conv_2d_transpose_4697396conv_2d_transpose_4697398*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv_2d_transpose_layer_call_and_return_conditional_losses_46969502+
)conv_2d_transpose/StatefulPartitionedCall?
lambda_7/PartitionedCallPartitionedCall2conv_2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_46972882
lambda_7/PartitionedCall?
Un_Normalize/PartitionedCallPartitionedCall!lambda_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_46973142
Un_Normalize/PartitionedCall?
IdentityIdentity%Un_Normalize/PartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall*^conv_2d_transpose/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2V
)conv_2d_transpose/StatefulPartitionedCall)conv_2d_transpose/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
.__inference_Un_Normalize_layer_call_fn_4700131
x
identity?
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_46973142
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:` \
=
_output_shapes+
):'???????????????????????????

_user_specified_namex
?.
?
D__inference_encoder_layer_call_and_return_conditional_losses_4696768

inputs
conv2d_4696734
conv2d_4696736
conv1d_4696740
conv1d_4696742
conv1d_1_4696745
conv1d_1_4696747
dense_4696752
dense_4696754
dense_1_4696757
dense_1_4696759
	z_4696762
	z_4696764
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?z/StatefulPartitionedCall?
Std_Normalize/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_46963762
Std_Normalize/PartitionedCall?
lambda/PartitionedCallPartitionedCall&Std_Normalize/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_46963962
lambda/PartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_4696734conv2d_4696736*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_46964202 
conv2d/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_46964462
lambda_1/PartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_4696740conv1d_4696742*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_46964752 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_4696745conv1d_1_4696747*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_46965072"
 conv1d_1/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_layer_call_and_return_conditional_losses_46963582#
!average_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall*average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_46965302
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4696752dense_4696754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_46965492
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4696757dense_1_4696759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_46965762!
dense_1/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0	z_4696762	z_4696764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_z_layer_call_and_return_conditional_losses_46966022
z/StatefulPartitionedCall?
IdentityIdentity"z/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4697781
x
encoder_4697730
encoder_4697732
encoder_4697734
encoder_4697736
encoder_4697738
encoder_4697740
encoder_4697742
encoder_4697744
encoder_4697746
encoder_4697748
encoder_4697750
encoder_4697752
decoder_4697755
decoder_4697757
decoder_4697759
decoder_4697761
decoder_4697763
decoder_4697765
decoder_4697767
decoder_4697769
decoder_4697771
decoder_4697773
decoder_4697775
decoder_4697777
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallxencoder_4697730encoder_4697732encoder_4697734encoder_4697736encoder_4697738encoder_4697740encoder_4697742encoder_4697744encoder_4697746encoder_4697748encoder_4697750encoder_4697752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_46967682!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_4697755decoder_4697757decoder_4697759decoder_4697761decoder_4697763decoder_4697765decoder_4697767decoder_4697769decoder_4697771decoder_4697773decoder_4697775decoder_4697777*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46974722!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d::::::::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:N J
+
_output_shapes
:?????????d

_user_specified_namex
?I
?
D__inference_encoder_layer_call_and_return_conditional_losses_4699201

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource$
 z_matmul_readvariableop_resource%
!z_biasadd_readvariableop_resource
identity?
Std_Normalize/sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
Std_Normalize/sub/y?
Std_Normalize/subSubinputsStd_Normalize/sub/y:output:0*
T0*+
_output_shapes
:?????????d2
Std_Normalize/sub?
Std_Normalize/truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
Std_Normalize/truediv/y?
Std_Normalize/truedivRealDivStd_Normalize/sub:z:0 Std_Normalize/truediv/y:output:0*
T0*+
_output_shapes
:?????????d2
Std_Normalize/truedivp
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/ExpandDims/dim?
lambda/ExpandDims
ExpandDimsStd_Normalize/truediv:z:0lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2
lambda/ExpandDims?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dlambda/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2
conv2d/BiasAddr

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2

conv2d/Elu?
lambda_1/SqueezeSqueezeconv2d/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2
lambda_1/Squeeze?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimslambda_1/Squeeze:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
conv1d/BiasAddn

conv1d/EluEluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2

conv1d/Elu?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Elu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2
conv1d_1/BiasAddt
conv1d_1/EluEluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^2
conv1d_1/Elu?
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dim?
average_pooling1d/ExpandDims
ExpandDimsconv1d_1/Elu:activations:0)average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2
average_pooling1d/ExpandDims?
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2
average_pooling1d/AvgPool?
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2
average_pooling1d/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
flatten/Const?
flatten/ReshapeReshape"average_pooling1d/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
dense/BiasAddg
	dense/EluEludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
	dense/Elu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Elu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddm
dense_1/EluEludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Elu?
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z/MatMul/ReadVariableOp?
z/MatMulMatMuldense_1/Elu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

z/MatMul?
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z/BiasAdd/ReadVariableOp?
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	z/BiasAddf
IdentityIdentityz/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d:::::::::::::S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
x
#__inference_z_layer_call_fn_4699826

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_z_layer_call_and_return_conditional_losses_46966022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_particle_autoencoder_layer_call_fn_4699044
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_46977812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????d

_user_specified_namex
?
?
2__inference_conv2d_transpose_layer_call_fn_4696863

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_46968532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
D__inference_decoder_layer_call_and_return_conditional_losses_4699573

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resourceN
Jconv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resourceE
Aconv1d_transpose_conv2d_transpose_biasadd_readvariableop_resourceR
Nconv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceI
Econv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource>
:conv_2d_transpose_conv2d_transpose_readvariableop_resource5
1conv_2d_transpose_biasadd_readvariableop_resource
identity??
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddm
dense_2/EluEludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Elu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Elu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
dense_3/BiasAddm
dense_3/EluEludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
dense_3/Elu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Elu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddn
dense_4/EluEludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/Elug
reshape/ShapeShapedense_4/Elu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_4/Elu:activations:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????/2
reshape/Reshapel
up_sampling1d/ConstConst*
_output_shapes
: *
dtype0*
value	B :/2
up_sampling1d/Const?
up_sampling1d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling1d/split/split_dim?	
up_sampling1d/splitSplit&up_sampling1d/split/split_dim:output:0reshape/Reshape:output:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split/2
up_sampling1d/splitx
up_sampling1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling1d/concat/axis?
up_sampling1d/concatConcatV2up_sampling1d/split:output:0up_sampling1d/split:output:0up_sampling1d/split:output:1up_sampling1d/split:output:1up_sampling1d/split:output:2up_sampling1d/split:output:2up_sampling1d/split:output:3up_sampling1d/split:output:3up_sampling1d/split:output:4up_sampling1d/split:output:4up_sampling1d/split:output:5up_sampling1d/split:output:5up_sampling1d/split:output:6up_sampling1d/split:output:6up_sampling1d/split:output:7up_sampling1d/split:output:7up_sampling1d/split:output:8up_sampling1d/split:output:8up_sampling1d/split:output:9up_sampling1d/split:output:9up_sampling1d/split:output:10up_sampling1d/split:output:10up_sampling1d/split:output:11up_sampling1d/split:output:11up_sampling1d/split:output:12up_sampling1d/split:output:12up_sampling1d/split:output:13up_sampling1d/split:output:13up_sampling1d/split:output:14up_sampling1d/split:output:14up_sampling1d/split:output:15up_sampling1d/split:output:15up_sampling1d/split:output:16up_sampling1d/split:output:16up_sampling1d/split:output:17up_sampling1d/split:output:17up_sampling1d/split:output:18up_sampling1d/split:output:18up_sampling1d/split:output:19up_sampling1d/split:output:19up_sampling1d/split:output:20up_sampling1d/split:output:20up_sampling1d/split:output:21up_sampling1d/split:output:21up_sampling1d/split:output:22up_sampling1d/split:output:22up_sampling1d/split:output:23up_sampling1d/split:output:23up_sampling1d/split:output:24up_sampling1d/split:output:24up_sampling1d/split:output:25up_sampling1d/split:output:25up_sampling1d/split:output:26up_sampling1d/split:output:26up_sampling1d/split:output:27up_sampling1d/split:output:27up_sampling1d/split:output:28up_sampling1d/split:output:28up_sampling1d/split:output:29up_sampling1d/split:output:29up_sampling1d/split:output:30up_sampling1d/split:output:30up_sampling1d/split:output:31up_sampling1d/split:output:31up_sampling1d/split:output:32up_sampling1d/split:output:32up_sampling1d/split:output:33up_sampling1d/split:output:33up_sampling1d/split:output:34up_sampling1d/split:output:34up_sampling1d/split:output:35up_sampling1d/split:output:35up_sampling1d/split:output:36up_sampling1d/split:output:36up_sampling1d/split:output:37up_sampling1d/split:output:37up_sampling1d/split:output:38up_sampling1d/split:output:38up_sampling1d/split:output:39up_sampling1d/split:output:39up_sampling1d/split:output:40up_sampling1d/split:output:40up_sampling1d/split:output:41up_sampling1d/split:output:41up_sampling1d/split:output:42up_sampling1d/split:output:42up_sampling1d/split:output:43up_sampling1d/split:output:43up_sampling1d/split:output:44up_sampling1d/split:output:44up_sampling1d/split:output:45up_sampling1d/split:output:45up_sampling1d/split:output:46up_sampling1d/split:output:46"up_sampling1d/concat/axis:output:0*
N^*
T0*+
_output_shapes
:?????????^2
up_sampling1d/concat?
(conv1d_transpose/lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(conv1d_transpose/lambda_2/ExpandDims/dim?
$conv1d_transpose/lambda_2/ExpandDims
ExpandDimsup_sampling1d/concat:output:01conv1d_transpose/lambda_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2&
$conv1d_transpose/lambda_2/ExpandDims?
'conv1d_transpose/conv2d_transpose/ShapeShape-conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*
_output_shapes
:2)
'conv1d_transpose/conv2d_transpose/Shape?
5conv1d_transpose/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5conv1d_transpose/conv2d_transpose/strided_slice/stack?
7conv1d_transpose/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv2d_transpose/strided_slice/stack_1?
7conv1d_transpose/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv2d_transpose/strided_slice/stack_2?
/conv1d_transpose/conv2d_transpose/strided_sliceStridedSlice0conv1d_transpose/conv2d_transpose/Shape:output:0>conv1d_transpose/conv2d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv2d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/conv1d_transpose/conv2d_transpose/strided_slice?
)conv1d_transpose/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2+
)conv1d_transpose/conv2d_transpose/stack/1?
)conv1d_transpose/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)conv1d_transpose/conv2d_transpose/stack/2?
)conv1d_transpose/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)conv1d_transpose/conv2d_transpose/stack/3?
'conv1d_transpose/conv2d_transpose/stackPack8conv1d_transpose/conv2d_transpose/strided_slice:output:02conv1d_transpose/conv2d_transpose/stack/1:output:02conv1d_transpose/conv2d_transpose/stack/2:output:02conv1d_transpose/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'conv1d_transpose/conv2d_transpose/stack?
7conv1d_transpose/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose/conv2d_transpose/strided_slice_1/stack?
9conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1?
9conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2?
1conv1d_transpose/conv2d_transpose/strided_slice_1StridedSlice0conv1d_transpose/conv2d_transpose/stack:output:0@conv1d_transpose/conv2d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv2d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1conv1d_transpose/conv2d_transpose/strided_slice_1?
Aconv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpJconv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02C
Aconv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp?
2conv1d_transpose/conv2d_transpose/conv2d_transposeConv2DBackpropInput0conv1d_transpose/conv2d_transpose/stack:output:0Iconv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0-conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
24
2conv1d_transpose/conv2d_transpose/conv2d_transpose?
8conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpAconv1d_transpose_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp?
)conv1d_transpose/conv2d_transpose/BiasAddBiasAdd;conv1d_transpose/conv2d_transpose/conv2d_transpose:output:0@conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`2+
)conv1d_transpose/conv2d_transpose/BiasAdd?
%conv1d_transpose/conv2d_transpose/EluElu2conv1d_transpose/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`2'
%conv1d_transpose/conv2d_transpose/Elu?
!conv1d_transpose/lambda_3/SqueezeSqueeze3conv1d_transpose/conv2d_transpose/Elu:activations:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims
2#
!conv1d_transpose/lambda_3/Squeeze?
*conv1d_transpose_1/lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*conv1d_transpose_1/lambda_4/ExpandDims/dim?
&conv1d_transpose_1/lambda_4/ExpandDims
ExpandDims*conv1d_transpose/lambda_3/Squeeze:output:03conv1d_transpose_1/lambda_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2(
&conv1d_transpose_1/lambda_4/ExpandDims?
+conv1d_transpose_1/conv2d_transpose_1/ShapeShape/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*
_output_shapes
:2-
+conv1d_transpose_1/conv2d_transpose_1/Shape?
9conv1d_transpose_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack?
;conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1?
;conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2?
3conv1d_transpose_1/conv2d_transpose_1/strided_sliceStridedSlice4conv1d_transpose_1/conv2d_transpose_1/Shape:output:0Bconv1d_transpose_1/conv2d_transpose_1/strided_slice/stack:output:0Dconv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1:output:0Dconv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3conv1d_transpose_1/conv2d_transpose_1/strided_slice?
-conv1d_transpose_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b2/
-conv1d_transpose_1/conv2d_transpose_1/stack/1?
-conv1d_transpose_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-conv1d_transpose_1/conv2d_transpose_1/stack/2?
-conv1d_transpose_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-conv1d_transpose_1/conv2d_transpose_1/stack/3?
+conv1d_transpose_1/conv2d_transpose_1/stackPack<conv1d_transpose_1/conv2d_transpose_1/strided_slice:output:06conv1d_transpose_1/conv2d_transpose_1/stack/1:output:06conv1d_transpose_1/conv2d_transpose_1/stack/2:output:06conv1d_transpose_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+conv1d_transpose_1/conv2d_transpose_1/stack?
;conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack?
=conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1?
=conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2?
5conv1d_transpose_1/conv2d_transpose_1/strided_slice_1StridedSlice4conv1d_transpose_1/conv2d_transpose_1/stack:output:0Dconv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack:output:0Fconv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Fconv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5conv1d_transpose_1/conv2d_transpose_1/strided_slice_1?
Econv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpNconv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02G
Econv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
6conv1d_transpose_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput4conv1d_transpose_1/conv2d_transpose_1/stack:output:0Mconv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
28
6conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose?
<conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpEconv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
-conv1d_transpose_1/conv2d_transpose_1/BiasAddBiasAdd?conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose:output:0Dconv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2/
-conv1d_transpose_1/conv2d_transpose_1/BiasAdd?
)conv1d_transpose_1/conv2d_transpose_1/EluElu6conv1d_transpose_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2+
)conv1d_transpose_1/conv2d_transpose_1/Elu?
#conv1d_transpose_1/lambda_5/SqueezeSqueeze7conv1d_transpose_1/conv2d_transpose_1/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2%
#conv1d_transpose_1/lambda_5/Squeezet
lambda_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_6/ExpandDims/dim?
lambda_6/ExpandDims
ExpandDims,conv1d_transpose_1/lambda_5/Squeeze:output:0 lambda_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
lambda_6/ExpandDims~
conv_2d_transpose/ShapeShapelambda_6/ExpandDims:output:0*
T0*
_output_shapes
:2
conv_2d_transpose/Shape?
%conv_2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%conv_2d_transpose/strided_slice/stack?
'conv_2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'conv_2d_transpose/strided_slice/stack_1?
'conv_2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'conv_2d_transpose/strided_slice/stack_2?
conv_2d_transpose/strided_sliceStridedSlice conv_2d_transpose/Shape:output:0.conv_2d_transpose/strided_slice/stack:output:00conv_2d_transpose/strided_slice/stack_1:output:00conv_2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
conv_2d_transpose/strided_slicex
conv_2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d2
conv_2d_transpose/stack/1x
conv_2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_2d_transpose/stack/2x
conv_2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_2d_transpose/stack/3?
conv_2d_transpose/stackPack(conv_2d_transpose/strided_slice:output:0"conv_2d_transpose/stack/1:output:0"conv_2d_transpose/stack/2:output:0"conv_2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_2d_transpose/stack?
'conv_2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv_2d_transpose/strided_slice_1/stack?
)conv_2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv_2d_transpose/strided_slice_1/stack_1?
)conv_2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv_2d_transpose/strided_slice_1/stack_2?
!conv_2d_transpose/strided_slice_1StridedSlice conv_2d_transpose/stack:output:00conv_2d_transpose/strided_slice_1/stack:output:02conv_2d_transpose/strided_slice_1/stack_1:output:02conv_2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv_2d_transpose/strided_slice_1?
1conv_2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp:conv_2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv_2d_transpose/conv2d_transpose/ReadVariableOp?
"conv_2d_transpose/conv2d_transposeConv2DBackpropInput conv_2d_transpose/stack:output:09conv_2d_transpose/conv2d_transpose/ReadVariableOp:value:0lambda_6/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2$
"conv_2d_transpose/conv2d_transpose?
(conv_2d_transpose/BiasAdd/ReadVariableOpReadVariableOp1conv_2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(conv_2d_transpose/BiasAdd/ReadVariableOp?
conv_2d_transpose/BiasAddBiasAdd+conv_2d_transpose/conv2d_transpose:output:00conv_2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2
conv_2d_transpose/BiasAdd?
lambda_7/SqueezeSqueeze"conv_2d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
2
lambda_7/Squeeze}
Un_Normalize/mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
Un_Normalize/mul/y?
Un_Normalize/mulMullambda_7/Squeeze:output:0Un_Normalize/mul/y:output:0*
T0*+
_output_shapes
:?????????d2
Un_Normalize/mul}
Un_Normalize/add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
Un_Normalize/add/y?
Un_Normalize/addAddV2Un_Normalize/mul:z:0Un_Normalize/add/y:output:0*
T0*+
_output_shapes
:?????????d2
Un_Normalize/addl
IdentityIdentityUn_Normalize/add:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_4696441

inputs
identityr
SqueezeSqueezeinputs*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????b:W S
/
_output_shapes
:?????????b
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_4699762

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
}
(__inference_conv2d_layer_call_fn_4699686

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_46964202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
E__inference_conv1d_1_layer_call_and_return_conditional_losses_4699747

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2	
BiasAddY
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:?????????^2
Elui
IdentityIdentityElu:activations:0*
T0*+
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????`:::S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?-
?
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_4697194

inputs?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identity?t
lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_4/ExpandDims/dim?
lambda_4/ExpandDims
ExpandDimsinputs lambda_4/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
lambda_4/ExpandDims?
conv2d_transpose_1/ShapeShapelambda_4/ExpandDims:output:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulv
conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/add/y?
conv2d_transpose_1/addAddV2conv2d_transpose_1/mul:z:0!conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/addz
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/add:z:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_2/stack?
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1?
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0lambda_4/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose_1/Elu?
lambda_5/SqueezeSqueeze$conv2d_transpose_1/Elu:activations:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
lambda_5/Squeezez
IdentityIdentitylambda_5/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
ɨ
?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698938
x1
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource>
:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource2
.encoder_conv1d_biasadd_readvariableop_resource@
<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_1_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.encoder_dense_1_matmul_readvariableop_resource3
/encoder_dense_1_biasadd_readvariableop_resource,
(encoder_z_matmul_readvariableop_resource-
)encoder_z_biasadd_readvariableop_resource2
.decoder_dense_2_matmul_readvariableop_resource3
/decoder_dense_2_biasadd_readvariableop_resource2
.decoder_dense_3_matmul_readvariableop_resource3
/decoder_dense_3_biasadd_readvariableop_resource2
.decoder_dense_4_matmul_readvariableop_resource3
/decoder_dense_4_biasadd_readvariableop_resourceV
Rdecoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resourceM
Idecoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resourceZ
Vdecoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceQ
Mdecoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resourceF
Bdecoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource=
9decoder_conv_2d_transpose_biasadd_readvariableop_resource
identity??
encoder/Std_Normalize/sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
encoder/Std_Normalize/sub/y?
encoder/Std_Normalize/subSubx$encoder/Std_Normalize/sub/y:output:0*
T0*+
_output_shapes
:?????????d2
encoder/Std_Normalize/sub?
encoder/Std_Normalize/truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2!
encoder/Std_Normalize/truediv/y?
encoder/Std_Normalize/truedivRealDivencoder/Std_Normalize/sub:z:0(encoder/Std_Normalize/truediv/y:output:0*
T0*+
_output_shapes
:?????????d2
encoder/Std_Normalize/truediv?
encoder/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
encoder/lambda/ExpandDims/dim?
encoder/lambda/ExpandDims
ExpandDims!encoder/Std_Normalize/truediv:z:0&encoder/lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2
encoder/lambda/ExpandDims?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2D"encoder/lambda/ExpandDims:output:0,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2
encoder/conv2d/BiasAdd?
encoder/conv2d/EluEluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2
encoder/conv2d/Elu?
encoder/lambda_1/SqueezeSqueeze encoder/conv2d/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2
encoder/lambda_1/Squeeze?
$encoder/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$encoder/conv1d/conv1d/ExpandDims/dim?
 encoder/conv1d/conv1d/ExpandDims
ExpandDims!encoder/lambda_1/Squeeze:output:0-encoder/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2"
 encoder/conv1d/conv1d/ExpandDims?
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype023
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
&encoder/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&encoder/conv1d/conv1d/ExpandDims_1/dim?
"encoder/conv1d/conv1d/ExpandDims_1
ExpandDims9encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0/encoder/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2$
"encoder/conv1d/conv1d/ExpandDims_1?
encoder/conv1d/conv1dConv2D)encoder/conv1d/conv1d/ExpandDims:output:0+encoder/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2
encoder/conv1d/conv1d?
encoder/conv1d/conv1d/SqueezeSqueezeencoder/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????2
encoder/conv1d/conv1d/Squeeze?
%encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv1d/BiasAdd/ReadVariableOp?
encoder/conv1d/BiasAddBiasAdd&encoder/conv1d/conv1d/Squeeze:output:0-encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
encoder/conv1d/BiasAdd?
encoder/conv1d/EluEluencoder/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
encoder/conv1d/Elu?
&encoder/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&encoder/conv1d_1/conv1d/ExpandDims/dim?
"encoder/conv1d_1/conv1d/ExpandDims
ExpandDims encoder/conv1d/Elu:activations:0/encoder/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2$
"encoder/conv1d_1/conv1d/ExpandDims?
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype025
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
(encoder/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_1/conv1d/ExpandDims_1/dim?
$encoder/conv1d_1/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2&
$encoder/conv1d_1/conv1d/ExpandDims_1?
encoder/conv1d_1/conv1dConv2D+encoder/conv1d_1/conv1d/ExpandDims:output:0-encoder/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2
encoder/conv1d_1/conv1d?
encoder/conv1d_1/conv1d/SqueezeSqueeze encoder/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????2!
encoder/conv1d_1/conv1d/Squeeze?
'encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'encoder/conv1d_1/BiasAdd/ReadVariableOp?
encoder/conv1d_1/BiasAddBiasAdd(encoder/conv1d_1/conv1d/Squeeze:output:0/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2
encoder/conv1d_1/BiasAdd?
encoder/conv1d_1/EluElu!encoder/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^2
encoder/conv1d_1/Elu?
(encoder/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(encoder/average_pooling1d/ExpandDims/dim?
$encoder/average_pooling1d/ExpandDims
ExpandDims"encoder/conv1d_1/Elu:activations:01encoder/average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2&
$encoder/average_pooling1d/ExpandDims?
!encoder/average_pooling1d/AvgPoolAvgPool-encoder/average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2#
!encoder/average_pooling1d/AvgPool?
!encoder/average_pooling1d/SqueezeSqueeze*encoder/average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2#
!encoder/average_pooling1d/Squeeze
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape*encoder/average_pooling1d/Squeeze:output:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/BiasAdd
encoder/dense/EluEluencoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/Elu?
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02'
%encoder/dense_1/MatMul/ReadVariableOp?
encoder/dense_1/MatMulMatMulencoder/dense/Elu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/MatMul?
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_1/BiasAdd/ReadVariableOp?
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/BiasAdd?
encoder/dense_1/EluElu encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/Elu?
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
encoder/z/MatMul/ReadVariableOp?
encoder/z/MatMulMatMul!encoder/dense_1/Elu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z/MatMul?
 encoder/z/BiasAdd/ReadVariableOpReadVariableOp)encoder_z_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 encoder/z/BiasAdd/ReadVariableOp?
encoder/z/BiasAddBiasAddencoder/z/MatMul:product:0(encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z/BiasAdd?
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%decoder/dense_2/MatMul/ReadVariableOp?
decoder/dense_2/MatMulMatMulencoder/z/BiasAdd:output:0-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/MatMul?
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&decoder/dense_2/BiasAdd/ReadVariableOp?
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/BiasAdd?
decoder/dense_2/EluElu decoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/Elu?
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02'
%decoder/dense_3/MatMul/ReadVariableOp?
decoder/dense_3/MatMulMatMul!decoder/dense_2/Elu:activations:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/MatMul?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02(
&decoder/dense_3/BiasAdd/ReadVariableOp?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/BiasAdd?
decoder/dense_3/EluElu decoder/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/Elu?
%decoder/dense_4/MatMul/ReadVariableOpReadVariableOp.decoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02'
%decoder/dense_4/MatMul/ReadVariableOp?
decoder/dense_4/MatMulMatMul!decoder/dense_3/Elu:activations:0-decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/MatMul?
&decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_4/BiasAdd/ReadVariableOp?
decoder/dense_4/BiasAddBiasAdd decoder/dense_4/MatMul:product:0.decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/BiasAdd?
decoder/dense_4/EluElu decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/Elu
decoder/reshape/ShapeShape!decoder/dense_4/Elu:activations:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshape!decoder/dense_4/Elu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????/2
decoder/reshape/Reshape|
decoder/up_sampling1d/ConstConst*
_output_shapes
: *
dtype0*
value	B :/2
decoder/up_sampling1d/Const?
%decoder/up_sampling1d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%decoder/up_sampling1d/split/split_dim?

decoder/up_sampling1d/splitSplit.decoder/up_sampling1d/split/split_dim:output:0 decoder/reshape/Reshape:output:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split/2
decoder/up_sampling1d/split?
!decoder/up_sampling1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/up_sampling1d/concat/axis?
decoder/up_sampling1d/concatConcatV2$decoder/up_sampling1d/split:output:0$decoder/up_sampling1d/split:output:0$decoder/up_sampling1d/split:output:1$decoder/up_sampling1d/split:output:1$decoder/up_sampling1d/split:output:2$decoder/up_sampling1d/split:output:2$decoder/up_sampling1d/split:output:3$decoder/up_sampling1d/split:output:3$decoder/up_sampling1d/split:output:4$decoder/up_sampling1d/split:output:4$decoder/up_sampling1d/split:output:5$decoder/up_sampling1d/split:output:5$decoder/up_sampling1d/split:output:6$decoder/up_sampling1d/split:output:6$decoder/up_sampling1d/split:output:7$decoder/up_sampling1d/split:output:7$decoder/up_sampling1d/split:output:8$decoder/up_sampling1d/split:output:8$decoder/up_sampling1d/split:output:9$decoder/up_sampling1d/split:output:9%decoder/up_sampling1d/split:output:10%decoder/up_sampling1d/split:output:10%decoder/up_sampling1d/split:output:11%decoder/up_sampling1d/split:output:11%decoder/up_sampling1d/split:output:12%decoder/up_sampling1d/split:output:12%decoder/up_sampling1d/split:output:13%decoder/up_sampling1d/split:output:13%decoder/up_sampling1d/split:output:14%decoder/up_sampling1d/split:output:14%decoder/up_sampling1d/split:output:15%decoder/up_sampling1d/split:output:15%decoder/up_sampling1d/split:output:16%decoder/up_sampling1d/split:output:16%decoder/up_sampling1d/split:output:17%decoder/up_sampling1d/split:output:17%decoder/up_sampling1d/split:output:18%decoder/up_sampling1d/split:output:18%decoder/up_sampling1d/split:output:19%decoder/up_sampling1d/split:output:19%decoder/up_sampling1d/split:output:20%decoder/up_sampling1d/split:output:20%decoder/up_sampling1d/split:output:21%decoder/up_sampling1d/split:output:21%decoder/up_sampling1d/split:output:22%decoder/up_sampling1d/split:output:22%decoder/up_sampling1d/split:output:23%decoder/up_sampling1d/split:output:23%decoder/up_sampling1d/split:output:24%decoder/up_sampling1d/split:output:24%decoder/up_sampling1d/split:output:25%decoder/up_sampling1d/split:output:25%decoder/up_sampling1d/split:output:26%decoder/up_sampling1d/split:output:26%decoder/up_sampling1d/split:output:27%decoder/up_sampling1d/split:output:27%decoder/up_sampling1d/split:output:28%decoder/up_sampling1d/split:output:28%decoder/up_sampling1d/split:output:29%decoder/up_sampling1d/split:output:29%decoder/up_sampling1d/split:output:30%decoder/up_sampling1d/split:output:30%decoder/up_sampling1d/split:output:31%decoder/up_sampling1d/split:output:31%decoder/up_sampling1d/split:output:32%decoder/up_sampling1d/split:output:32%decoder/up_sampling1d/split:output:33%decoder/up_sampling1d/split:output:33%decoder/up_sampling1d/split:output:34%decoder/up_sampling1d/split:output:34%decoder/up_sampling1d/split:output:35%decoder/up_sampling1d/split:output:35%decoder/up_sampling1d/split:output:36%decoder/up_sampling1d/split:output:36%decoder/up_sampling1d/split:output:37%decoder/up_sampling1d/split:output:37%decoder/up_sampling1d/split:output:38%decoder/up_sampling1d/split:output:38%decoder/up_sampling1d/split:output:39%decoder/up_sampling1d/split:output:39%decoder/up_sampling1d/split:output:40%decoder/up_sampling1d/split:output:40%decoder/up_sampling1d/split:output:41%decoder/up_sampling1d/split:output:41%decoder/up_sampling1d/split:output:42%decoder/up_sampling1d/split:output:42%decoder/up_sampling1d/split:output:43%decoder/up_sampling1d/split:output:43%decoder/up_sampling1d/split:output:44%decoder/up_sampling1d/split:output:44%decoder/up_sampling1d/split:output:45%decoder/up_sampling1d/split:output:45%decoder/up_sampling1d/split:output:46%decoder/up_sampling1d/split:output:46*decoder/up_sampling1d/concat/axis:output:0*
N^*
T0*+
_output_shapes
:?????????^2
decoder/up_sampling1d/concat?
0decoder/conv1d_transpose/lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0decoder/conv1d_transpose/lambda_2/ExpandDims/dim?
,decoder/conv1d_transpose/lambda_2/ExpandDims
ExpandDims%decoder/up_sampling1d/concat:output:09decoder/conv1d_transpose/lambda_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2.
,decoder/conv1d_transpose/lambda_2/ExpandDims?
/decoder/conv1d_transpose/conv2d_transpose/ShapeShape5decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*
_output_shapes
:21
/decoder/conv1d_transpose/conv2d_transpose/Shape?
=decoder/conv1d_transpose/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2?
7decoder/conv1d_transpose/conv2d_transpose/strided_sliceStridedSlice8decoder/conv1d_transpose/conv2d_transpose/Shape:output:0Fdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7decoder/conv1d_transpose/conv2d_transpose/strided_slice?
1decoder/conv1d_transpose/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`23
1decoder/conv1d_transpose/conv2d_transpose/stack/1?
1decoder/conv1d_transpose/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :23
1decoder/conv1d_transpose/conv2d_transpose/stack/2?
1decoder/conv1d_transpose/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :23
1decoder/conv1d_transpose/conv2d_transpose/stack/3?
/decoder/conv1d_transpose/conv2d_transpose/stackPack@decoder/conv1d_transpose/conv2d_transpose/strided_slice:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/1:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/2:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:21
/decoder/conv1d_transpose/conv2d_transpose/stack?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack?
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1?
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2?
9decoder/conv1d_transpose/conv2d_transpose/strided_slice_1StridedSlice8decoder/conv1d_transpose/conv2d_transpose/stack:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack:output:0Jdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1:output:0Jdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9decoder/conv1d_transpose/conv2d_transpose/strided_slice_1?
Idecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpRdecoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02K
Idecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp?
:decoder/conv1d_transpose/conv2d_transpose/conv2d_transposeConv2DBackpropInput8decoder/conv1d_transpose/conv2d_transpose/stack:output:0Qdecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:05decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2<
:decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose?
@decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpIdecoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp?
1decoder/conv1d_transpose/conv2d_transpose/BiasAddBiasAddCdecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose:output:0Hdecoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`23
1decoder/conv1d_transpose/conv2d_transpose/BiasAdd?
-decoder/conv1d_transpose/conv2d_transpose/EluElu:decoder/conv1d_transpose/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`2/
-decoder/conv1d_transpose/conv2d_transpose/Elu?
)decoder/conv1d_transpose/lambda_3/SqueezeSqueeze;decoder/conv1d_transpose/conv2d_transpose/Elu:activations:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims
2+
)decoder/conv1d_transpose/lambda_3/Squeeze?
2decoder/conv1d_transpose_1/lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim?
.decoder/conv1d_transpose_1/lambda_4/ExpandDims
ExpandDims2decoder/conv1d_transpose/lambda_3/Squeeze:output:0;decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`20
.decoder/conv1d_transpose_1/lambda_4/ExpandDims?
3decoder/conv1d_transpose_1/conv2d_transpose_1/ShapeShape7decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*
_output_shapes
:25
3decoder/conv1d_transpose_1/conv2d_transpose_1/Shape?
Adecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Adecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2?
;decoder/conv1d_transpose_1/conv2d_transpose_1/strided_sliceStridedSlice<decoder/conv1d_transpose_1/conv2d_transpose_1/Shape:output:0Jdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3?
3decoder/conv1d_transpose_1/conv2d_transpose_1/stackPackDdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:25
3decoder/conv1d_transpose_1/conv2d_transpose_1/stack?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack?
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1?
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2?
=decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1StridedSlice<decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack:output:0Ndecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ndecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1?
Mdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpVdecoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02O
Mdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
>decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput<decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0Udecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:07decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2@
>decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose?
Ddecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpMdecoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02F
Ddecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
5decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAddBiasAddGdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b27
5decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd?
1decoder/conv1d_transpose_1/conv2d_transpose_1/EluElu>decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b23
1decoder/conv1d_transpose_1/conv2d_transpose_1/Elu?
+decoder/conv1d_transpose_1/lambda_5/SqueezeSqueeze?decoder/conv1d_transpose_1/conv2d_transpose_1/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2-
+decoder/conv1d_transpose_1/lambda_5/Squeeze?
decoder/lambda_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
decoder/lambda_6/ExpandDims/dim?
decoder/lambda_6/ExpandDims
ExpandDims4decoder/conv1d_transpose_1/lambda_5/Squeeze:output:0(decoder/lambda_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
decoder/lambda_6/ExpandDims?
decoder/conv_2d_transpose/ShapeShape$decoder/lambda_6/ExpandDims:output:0*
T0*
_output_shapes
:2!
decoder/conv_2d_transpose/Shape?
-decoder/conv_2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-decoder/conv_2d_transpose/strided_slice/stack?
/decoder/conv_2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/decoder/conv_2d_transpose/strided_slice/stack_1?
/decoder/conv_2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/decoder/conv_2d_transpose/strided_slice/stack_2?
'decoder/conv_2d_transpose/strided_sliceStridedSlice(decoder/conv_2d_transpose/Shape:output:06decoder/conv_2d_transpose/strided_slice/stack:output:08decoder/conv_2d_transpose/strided_slice/stack_1:output:08decoder/conv_2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'decoder/conv_2d_transpose/strided_slice?
!decoder/conv_2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d2#
!decoder/conv_2d_transpose/stack/1?
!decoder/conv_2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/conv_2d_transpose/stack/2?
!decoder/conv_2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/conv_2d_transpose/stack/3?
decoder/conv_2d_transpose/stackPack0decoder/conv_2d_transpose/strided_slice:output:0*decoder/conv_2d_transpose/stack/1:output:0*decoder/conv_2d_transpose/stack/2:output:0*decoder/conv_2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/conv_2d_transpose/stack?
/decoder/conv_2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv_2d_transpose/strided_slice_1/stack?
1decoder/conv_2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv_2d_transpose/strided_slice_1/stack_1?
1decoder/conv_2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv_2d_transpose/strided_slice_1/stack_2?
)decoder/conv_2d_transpose/strided_slice_1StridedSlice(decoder/conv_2d_transpose/stack:output:08decoder/conv_2d_transpose/strided_slice_1/stack:output:0:decoder/conv_2d_transpose/strided_slice_1/stack_1:output:0:decoder/conv_2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv_2d_transpose/strided_slice_1?
9decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpBdecoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02;
9decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp?
*decoder/conv_2d_transpose/conv2d_transposeConv2DBackpropInput(decoder/conv_2d_transpose/stack:output:0Adecoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp:value:0$decoder/lambda_6/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2,
*decoder/conv_2d_transpose/conv2d_transpose?
0decoder/conv_2d_transpose/BiasAdd/ReadVariableOpReadVariableOp9decoder_conv_2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0decoder/conv_2d_transpose/BiasAdd/ReadVariableOp?
!decoder/conv_2d_transpose/BiasAddBiasAdd3decoder/conv_2d_transpose/conv2d_transpose:output:08decoder/conv_2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2#
!decoder/conv_2d_transpose/BiasAdd?
decoder/lambda_7/SqueezeSqueeze*decoder/conv_2d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
2
decoder/lambda_7/Squeeze?
decoder/Un_Normalize/mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
decoder/Un_Normalize/mul/y?
decoder/Un_Normalize/mulMul!decoder/lambda_7/Squeeze:output:0#decoder/Un_Normalize/mul/y:output:0*
T0*+
_output_shapes
:?????????d2
decoder/Un_Normalize/mul?
decoder/Un_Normalize/add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
decoder/Un_Normalize/add/y?
decoder/Un_Normalize/addAddV2decoder/Un_Normalize/mul:z:0#decoder/Un_Normalize/add/y:output:0*
T0*+
_output_shapes
:?????????d2
decoder/Un_Normalize/addt
IdentityIdentitydecoder/Un_Normalize/add:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d:::::::::::::::::::::::::N J
+
_output_shapes
:?????????d

_user_specified_namex
?
`
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_4700126
x
identityc
mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
mul/yc
mulMulxmul/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
mulc
add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
add/yk
addAddV2mul:z:0add/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
addh
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:` \
=
_output_shapes+
):'???????????????????????????

_user_specified_namex
?
D
(__inference_lambda_layer_call_fn_4699666

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_46963962
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
E__inference_lambda_7_layer_call_and_return_conditional_losses_4700103

inputs
identity?
SqueezeSqueezeinputs*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?,
?
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_4699972

inputs=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource
identity?t
lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_2/ExpandDims/dim?
lambda_2/ExpandDims
ExpandDimsinputs lambda_2/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
lambda_2/ExpandDims|
conv2d_transpose/ShapeShapelambda_2/ExpandDims:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulr
conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/add/y?
conv2d_transpose/addAddV2conv2d_transpose/mul:z:0conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/addv
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/add:z:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_2/stack?
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1?
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0lambda_2/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose/BiasAdd?
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose/Elu?
lambda_3/SqueezeSqueeze"conv2d_transpose/Elu:activations:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
lambda_3/Squeezez
IdentityIdentitylambda_3/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'???????????????????????????:::e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
)__inference_decoder_layer_call_fn_4699631

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46974722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
)__inference_encoder_layer_call_fn_4696727
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_46967002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????d
'
_user_specified_nameencoder_input
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_4699650

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
F
*__inference_lambda_1_layer_call_fn_4699701

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_46964412
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????b:W S
/
_output_shapes
:?????????b
 
_user_specified_nameinputs
?
?
D__inference_dense_2_layer_call_and_return_conditional_losses_4699837

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ۨ
?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698390
input_11
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource>
:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource2
.encoder_conv1d_biasadd_readvariableop_resource@
<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_1_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.encoder_dense_1_matmul_readvariableop_resource3
/encoder_dense_1_biasadd_readvariableop_resource,
(encoder_z_matmul_readvariableop_resource-
)encoder_z_biasadd_readvariableop_resource2
.decoder_dense_2_matmul_readvariableop_resource3
/decoder_dense_2_biasadd_readvariableop_resource2
.decoder_dense_3_matmul_readvariableop_resource3
/decoder_dense_3_biasadd_readvariableop_resource2
.decoder_dense_4_matmul_readvariableop_resource3
/decoder_dense_4_biasadd_readvariableop_resourceV
Rdecoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resourceM
Idecoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resourceZ
Vdecoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceQ
Mdecoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resourceF
Bdecoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource=
9decoder_conv_2d_transpose_biasadd_readvariableop_resource
identity??
encoder/Std_Normalize/sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
encoder/Std_Normalize/sub/y?
encoder/Std_Normalize/subSubinput_1$encoder/Std_Normalize/sub/y:output:0*
T0*+
_output_shapes
:?????????d2
encoder/Std_Normalize/sub?
encoder/Std_Normalize/truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2!
encoder/Std_Normalize/truediv/y?
encoder/Std_Normalize/truedivRealDivencoder/Std_Normalize/sub:z:0(encoder/Std_Normalize/truediv/y:output:0*
T0*+
_output_shapes
:?????????d2
encoder/Std_Normalize/truediv?
encoder/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
encoder/lambda/ExpandDims/dim?
encoder/lambda/ExpandDims
ExpandDims!encoder/Std_Normalize/truediv:z:0&encoder/lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2
encoder/lambda/ExpandDims?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2D"encoder/lambda/ExpandDims:output:0,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2
encoder/conv2d/BiasAdd?
encoder/conv2d/EluEluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2
encoder/conv2d/Elu?
encoder/lambda_1/SqueezeSqueeze encoder/conv2d/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2
encoder/lambda_1/Squeeze?
$encoder/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$encoder/conv1d/conv1d/ExpandDims/dim?
 encoder/conv1d/conv1d/ExpandDims
ExpandDims!encoder/lambda_1/Squeeze:output:0-encoder/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2"
 encoder/conv1d/conv1d/ExpandDims?
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype023
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
&encoder/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&encoder/conv1d/conv1d/ExpandDims_1/dim?
"encoder/conv1d/conv1d/ExpandDims_1
ExpandDims9encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0/encoder/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2$
"encoder/conv1d/conv1d/ExpandDims_1?
encoder/conv1d/conv1dConv2D)encoder/conv1d/conv1d/ExpandDims:output:0+encoder/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2
encoder/conv1d/conv1d?
encoder/conv1d/conv1d/SqueezeSqueezeencoder/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????2
encoder/conv1d/conv1d/Squeeze?
%encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv1d/BiasAdd/ReadVariableOp?
encoder/conv1d/BiasAddBiasAdd&encoder/conv1d/conv1d/Squeeze:output:0-encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
encoder/conv1d/BiasAdd?
encoder/conv1d/EluEluencoder/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
encoder/conv1d/Elu?
&encoder/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&encoder/conv1d_1/conv1d/ExpandDims/dim?
"encoder/conv1d_1/conv1d/ExpandDims
ExpandDims encoder/conv1d/Elu:activations:0/encoder/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2$
"encoder/conv1d_1/conv1d/ExpandDims?
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype025
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
(encoder/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_1/conv1d/ExpandDims_1/dim?
$encoder/conv1d_1/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2&
$encoder/conv1d_1/conv1d/ExpandDims_1?
encoder/conv1d_1/conv1dConv2D+encoder/conv1d_1/conv1d/ExpandDims:output:0-encoder/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2
encoder/conv1d_1/conv1d?
encoder/conv1d_1/conv1d/SqueezeSqueeze encoder/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????2!
encoder/conv1d_1/conv1d/Squeeze?
'encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'encoder/conv1d_1/BiasAdd/ReadVariableOp?
encoder/conv1d_1/BiasAddBiasAdd(encoder/conv1d_1/conv1d/Squeeze:output:0/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2
encoder/conv1d_1/BiasAdd?
encoder/conv1d_1/EluElu!encoder/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^2
encoder/conv1d_1/Elu?
(encoder/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(encoder/average_pooling1d/ExpandDims/dim?
$encoder/average_pooling1d/ExpandDims
ExpandDims"encoder/conv1d_1/Elu:activations:01encoder/average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2&
$encoder/average_pooling1d/ExpandDims?
!encoder/average_pooling1d/AvgPoolAvgPool-encoder/average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2#
!encoder/average_pooling1d/AvgPool?
!encoder/average_pooling1d/SqueezeSqueeze*encoder/average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2#
!encoder/average_pooling1d/Squeeze
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape*encoder/average_pooling1d/Squeeze:output:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/BiasAdd
encoder/dense/EluEluencoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/Elu?
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02'
%encoder/dense_1/MatMul/ReadVariableOp?
encoder/dense_1/MatMulMatMulencoder/dense/Elu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/MatMul?
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_1/BiasAdd/ReadVariableOp?
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/BiasAdd?
encoder/dense_1/EluElu encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/Elu?
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
encoder/z/MatMul/ReadVariableOp?
encoder/z/MatMulMatMul!encoder/dense_1/Elu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z/MatMul?
 encoder/z/BiasAdd/ReadVariableOpReadVariableOp)encoder_z_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 encoder/z/BiasAdd/ReadVariableOp?
encoder/z/BiasAddBiasAddencoder/z/MatMul:product:0(encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z/BiasAdd?
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%decoder/dense_2/MatMul/ReadVariableOp?
decoder/dense_2/MatMulMatMulencoder/z/BiasAdd:output:0-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/MatMul?
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&decoder/dense_2/BiasAdd/ReadVariableOp?
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/BiasAdd?
decoder/dense_2/EluElu decoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/Elu?
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02'
%decoder/dense_3/MatMul/ReadVariableOp?
decoder/dense_3/MatMulMatMul!decoder/dense_2/Elu:activations:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/MatMul?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02(
&decoder/dense_3/BiasAdd/ReadVariableOp?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/BiasAdd?
decoder/dense_3/EluElu decoder/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/Elu?
%decoder/dense_4/MatMul/ReadVariableOpReadVariableOp.decoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02'
%decoder/dense_4/MatMul/ReadVariableOp?
decoder/dense_4/MatMulMatMul!decoder/dense_3/Elu:activations:0-decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/MatMul?
&decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_4/BiasAdd/ReadVariableOp?
decoder/dense_4/BiasAddBiasAdd decoder/dense_4/MatMul:product:0.decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/BiasAdd?
decoder/dense_4/EluElu decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/Elu
decoder/reshape/ShapeShape!decoder/dense_4/Elu:activations:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshape!decoder/dense_4/Elu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????/2
decoder/reshape/Reshape|
decoder/up_sampling1d/ConstConst*
_output_shapes
: *
dtype0*
value	B :/2
decoder/up_sampling1d/Const?
%decoder/up_sampling1d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%decoder/up_sampling1d/split/split_dim?

decoder/up_sampling1d/splitSplit.decoder/up_sampling1d/split/split_dim:output:0 decoder/reshape/Reshape:output:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split/2
decoder/up_sampling1d/split?
!decoder/up_sampling1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/up_sampling1d/concat/axis?
decoder/up_sampling1d/concatConcatV2$decoder/up_sampling1d/split:output:0$decoder/up_sampling1d/split:output:0$decoder/up_sampling1d/split:output:1$decoder/up_sampling1d/split:output:1$decoder/up_sampling1d/split:output:2$decoder/up_sampling1d/split:output:2$decoder/up_sampling1d/split:output:3$decoder/up_sampling1d/split:output:3$decoder/up_sampling1d/split:output:4$decoder/up_sampling1d/split:output:4$decoder/up_sampling1d/split:output:5$decoder/up_sampling1d/split:output:5$decoder/up_sampling1d/split:output:6$decoder/up_sampling1d/split:output:6$decoder/up_sampling1d/split:output:7$decoder/up_sampling1d/split:output:7$decoder/up_sampling1d/split:output:8$decoder/up_sampling1d/split:output:8$decoder/up_sampling1d/split:output:9$decoder/up_sampling1d/split:output:9%decoder/up_sampling1d/split:output:10%decoder/up_sampling1d/split:output:10%decoder/up_sampling1d/split:output:11%decoder/up_sampling1d/split:output:11%decoder/up_sampling1d/split:output:12%decoder/up_sampling1d/split:output:12%decoder/up_sampling1d/split:output:13%decoder/up_sampling1d/split:output:13%decoder/up_sampling1d/split:output:14%decoder/up_sampling1d/split:output:14%decoder/up_sampling1d/split:output:15%decoder/up_sampling1d/split:output:15%decoder/up_sampling1d/split:output:16%decoder/up_sampling1d/split:output:16%decoder/up_sampling1d/split:output:17%decoder/up_sampling1d/split:output:17%decoder/up_sampling1d/split:output:18%decoder/up_sampling1d/split:output:18%decoder/up_sampling1d/split:output:19%decoder/up_sampling1d/split:output:19%decoder/up_sampling1d/split:output:20%decoder/up_sampling1d/split:output:20%decoder/up_sampling1d/split:output:21%decoder/up_sampling1d/split:output:21%decoder/up_sampling1d/split:output:22%decoder/up_sampling1d/split:output:22%decoder/up_sampling1d/split:output:23%decoder/up_sampling1d/split:output:23%decoder/up_sampling1d/split:output:24%decoder/up_sampling1d/split:output:24%decoder/up_sampling1d/split:output:25%decoder/up_sampling1d/split:output:25%decoder/up_sampling1d/split:output:26%decoder/up_sampling1d/split:output:26%decoder/up_sampling1d/split:output:27%decoder/up_sampling1d/split:output:27%decoder/up_sampling1d/split:output:28%decoder/up_sampling1d/split:output:28%decoder/up_sampling1d/split:output:29%decoder/up_sampling1d/split:output:29%decoder/up_sampling1d/split:output:30%decoder/up_sampling1d/split:output:30%decoder/up_sampling1d/split:output:31%decoder/up_sampling1d/split:output:31%decoder/up_sampling1d/split:output:32%decoder/up_sampling1d/split:output:32%decoder/up_sampling1d/split:output:33%decoder/up_sampling1d/split:output:33%decoder/up_sampling1d/split:output:34%decoder/up_sampling1d/split:output:34%decoder/up_sampling1d/split:output:35%decoder/up_sampling1d/split:output:35%decoder/up_sampling1d/split:output:36%decoder/up_sampling1d/split:output:36%decoder/up_sampling1d/split:output:37%decoder/up_sampling1d/split:output:37%decoder/up_sampling1d/split:output:38%decoder/up_sampling1d/split:output:38%decoder/up_sampling1d/split:output:39%decoder/up_sampling1d/split:output:39%decoder/up_sampling1d/split:output:40%decoder/up_sampling1d/split:output:40%decoder/up_sampling1d/split:output:41%decoder/up_sampling1d/split:output:41%decoder/up_sampling1d/split:output:42%decoder/up_sampling1d/split:output:42%decoder/up_sampling1d/split:output:43%decoder/up_sampling1d/split:output:43%decoder/up_sampling1d/split:output:44%decoder/up_sampling1d/split:output:44%decoder/up_sampling1d/split:output:45%decoder/up_sampling1d/split:output:45%decoder/up_sampling1d/split:output:46%decoder/up_sampling1d/split:output:46*decoder/up_sampling1d/concat/axis:output:0*
N^*
T0*+
_output_shapes
:?????????^2
decoder/up_sampling1d/concat?
0decoder/conv1d_transpose/lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0decoder/conv1d_transpose/lambda_2/ExpandDims/dim?
,decoder/conv1d_transpose/lambda_2/ExpandDims
ExpandDims%decoder/up_sampling1d/concat:output:09decoder/conv1d_transpose/lambda_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2.
,decoder/conv1d_transpose/lambda_2/ExpandDims?
/decoder/conv1d_transpose/conv2d_transpose/ShapeShape5decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*
_output_shapes
:21
/decoder/conv1d_transpose/conv2d_transpose/Shape?
=decoder/conv1d_transpose/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2?
7decoder/conv1d_transpose/conv2d_transpose/strided_sliceStridedSlice8decoder/conv1d_transpose/conv2d_transpose/Shape:output:0Fdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7decoder/conv1d_transpose/conv2d_transpose/strided_slice?
1decoder/conv1d_transpose/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`23
1decoder/conv1d_transpose/conv2d_transpose/stack/1?
1decoder/conv1d_transpose/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :23
1decoder/conv1d_transpose/conv2d_transpose/stack/2?
1decoder/conv1d_transpose/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :23
1decoder/conv1d_transpose/conv2d_transpose/stack/3?
/decoder/conv1d_transpose/conv2d_transpose/stackPack@decoder/conv1d_transpose/conv2d_transpose/strided_slice:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/1:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/2:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:21
/decoder/conv1d_transpose/conv2d_transpose/stack?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack?
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1?
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2?
9decoder/conv1d_transpose/conv2d_transpose/strided_slice_1StridedSlice8decoder/conv1d_transpose/conv2d_transpose/stack:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack:output:0Jdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1:output:0Jdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9decoder/conv1d_transpose/conv2d_transpose/strided_slice_1?
Idecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpRdecoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02K
Idecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp?
:decoder/conv1d_transpose/conv2d_transpose/conv2d_transposeConv2DBackpropInput8decoder/conv1d_transpose/conv2d_transpose/stack:output:0Qdecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:05decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2<
:decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose?
@decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpIdecoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp?
1decoder/conv1d_transpose/conv2d_transpose/BiasAddBiasAddCdecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose:output:0Hdecoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`23
1decoder/conv1d_transpose/conv2d_transpose/BiasAdd?
-decoder/conv1d_transpose/conv2d_transpose/EluElu:decoder/conv1d_transpose/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`2/
-decoder/conv1d_transpose/conv2d_transpose/Elu?
)decoder/conv1d_transpose/lambda_3/SqueezeSqueeze;decoder/conv1d_transpose/conv2d_transpose/Elu:activations:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims
2+
)decoder/conv1d_transpose/lambda_3/Squeeze?
2decoder/conv1d_transpose_1/lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim?
.decoder/conv1d_transpose_1/lambda_4/ExpandDims
ExpandDims2decoder/conv1d_transpose/lambda_3/Squeeze:output:0;decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`20
.decoder/conv1d_transpose_1/lambda_4/ExpandDims?
3decoder/conv1d_transpose_1/conv2d_transpose_1/ShapeShape7decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*
_output_shapes
:25
3decoder/conv1d_transpose_1/conv2d_transpose_1/Shape?
Adecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Adecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2?
;decoder/conv1d_transpose_1/conv2d_transpose_1/strided_sliceStridedSlice<decoder/conv1d_transpose_1/conv2d_transpose_1/Shape:output:0Jdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3?
3decoder/conv1d_transpose_1/conv2d_transpose_1/stackPackDdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:25
3decoder/conv1d_transpose_1/conv2d_transpose_1/stack?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack?
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1?
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2?
=decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1StridedSlice<decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack:output:0Ndecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ndecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1?
Mdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpVdecoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02O
Mdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
>decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput<decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0Udecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:07decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2@
>decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose?
Ddecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpMdecoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02F
Ddecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
5decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAddBiasAddGdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b27
5decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd?
1decoder/conv1d_transpose_1/conv2d_transpose_1/EluElu>decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b23
1decoder/conv1d_transpose_1/conv2d_transpose_1/Elu?
+decoder/conv1d_transpose_1/lambda_5/SqueezeSqueeze?decoder/conv1d_transpose_1/conv2d_transpose_1/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2-
+decoder/conv1d_transpose_1/lambda_5/Squeeze?
decoder/lambda_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
decoder/lambda_6/ExpandDims/dim?
decoder/lambda_6/ExpandDims
ExpandDims4decoder/conv1d_transpose_1/lambda_5/Squeeze:output:0(decoder/lambda_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
decoder/lambda_6/ExpandDims?
decoder/conv_2d_transpose/ShapeShape$decoder/lambda_6/ExpandDims:output:0*
T0*
_output_shapes
:2!
decoder/conv_2d_transpose/Shape?
-decoder/conv_2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-decoder/conv_2d_transpose/strided_slice/stack?
/decoder/conv_2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/decoder/conv_2d_transpose/strided_slice/stack_1?
/decoder/conv_2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/decoder/conv_2d_transpose/strided_slice/stack_2?
'decoder/conv_2d_transpose/strided_sliceStridedSlice(decoder/conv_2d_transpose/Shape:output:06decoder/conv_2d_transpose/strided_slice/stack:output:08decoder/conv_2d_transpose/strided_slice/stack_1:output:08decoder/conv_2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'decoder/conv_2d_transpose/strided_slice?
!decoder/conv_2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d2#
!decoder/conv_2d_transpose/stack/1?
!decoder/conv_2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/conv_2d_transpose/stack/2?
!decoder/conv_2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/conv_2d_transpose/stack/3?
decoder/conv_2d_transpose/stackPack0decoder/conv_2d_transpose/strided_slice:output:0*decoder/conv_2d_transpose/stack/1:output:0*decoder/conv_2d_transpose/stack/2:output:0*decoder/conv_2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/conv_2d_transpose/stack?
/decoder/conv_2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv_2d_transpose/strided_slice_1/stack?
1decoder/conv_2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv_2d_transpose/strided_slice_1/stack_1?
1decoder/conv_2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv_2d_transpose/strided_slice_1/stack_2?
)decoder/conv_2d_transpose/strided_slice_1StridedSlice(decoder/conv_2d_transpose/stack:output:08decoder/conv_2d_transpose/strided_slice_1/stack:output:0:decoder/conv_2d_transpose/strided_slice_1/stack_1:output:0:decoder/conv_2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv_2d_transpose/strided_slice_1?
9decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpBdecoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02;
9decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp?
*decoder/conv_2d_transpose/conv2d_transposeConv2DBackpropInput(decoder/conv_2d_transpose/stack:output:0Adecoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp:value:0$decoder/lambda_6/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2,
*decoder/conv_2d_transpose/conv2d_transpose?
0decoder/conv_2d_transpose/BiasAdd/ReadVariableOpReadVariableOp9decoder_conv_2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0decoder/conv_2d_transpose/BiasAdd/ReadVariableOp?
!decoder/conv_2d_transpose/BiasAddBiasAdd3decoder/conv_2d_transpose/conv2d_transpose:output:08decoder/conv_2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2#
!decoder/conv_2d_transpose/BiasAdd?
decoder/lambda_7/SqueezeSqueeze*decoder/conv_2d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
2
decoder/lambda_7/Squeeze?
decoder/Un_Normalize/mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
decoder/Un_Normalize/mul/y?
decoder/Un_Normalize/mulMul!decoder/lambda_7/Squeeze:output:0#decoder/Un_Normalize/mul/y:output:0*
T0*+
_output_shapes
:?????????d2
decoder/Un_Normalize/mul?
decoder/Un_Normalize/add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
decoder/Un_Normalize/add/y?
decoder/Un_Normalize/addAddV2decoder/Un_Normalize/mul:z:0#decoder/Un_Normalize/add/y:output:0*
T0*+
_output_shapes
:?????????d2
decoder/Un_Normalize/addt
IdentityIdentitydecoder/Un_Normalize/add:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d:::::::::::::::::::::::::T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1
?

f
J__inference_up_sampling1d_layer_call_and_return_conditional_losses_4696808

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDimsy
Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"            2
Tile/multiples?
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tilec
ConstConst*
_output_shapes
:*
dtype0*!
valueB"         2
ConstV
mulMulShape:output:0Const:output:0*
T0*
_output_shapes
:2
mul}
ReshapeReshapeTile:output:0mul:z:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Reshapez
IdentityIdentityReshape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
j
N__inference_average_pooling1d_layer_call_and_return_conditional_losses_4696358

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_layer_call_and_return_conditional_losses_4699778

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_lambda_layer_call_fn_4699661

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_46963902
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?!
 __inference__traced_save_4700385
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop'
#savev2_z_kernel_read_readvariableop%
!savev2_z_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableopG
Csavev2_conv1d_transpose_conv2d_transpose_kernel_read_readvariableopE
Asavev2_conv1d_transpose_conv2d_transpose_bias_read_readvariableopK
Gsavev2_conv1d_transpose_1_conv2d_transpose_1_kernel_read_readvariableopI
Esavev2_conv1d_transpose_1_conv2d_transpose_1_bias_read_readvariableop7
3savev2_conv_2d_transpose_kernel_read_readvariableop5
1savev2_conv_2d_transpose_bias_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop.
*savev2_adam_z_kernel_m_read_readvariableop,
(savev2_adam_z_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableopN
Jsavev2_adam_conv1d_transpose_conv2d_transpose_kernel_m_read_readvariableopL
Hsavev2_adam_conv1d_transpose_conv2d_transpose_bias_m_read_readvariableopR
Nsavev2_adam_conv1d_transpose_1_conv2d_transpose_1_kernel_m_read_readvariableopP
Lsavev2_adam_conv1d_transpose_1_conv2d_transpose_1_bias_m_read_readvariableop>
:savev2_adam_conv_2d_transpose_kernel_m_read_readvariableop<
8savev2_adam_conv_2d_transpose_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop.
*savev2_adam_z_kernel_v_read_readvariableop,
(savev2_adam_z_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableopN
Jsavev2_adam_conv1d_transpose_conv2d_transpose_kernel_v_read_readvariableopL
Hsavev2_adam_conv1d_transpose_conv2d_transpose_bias_v_read_readvariableopR
Nsavev2_adam_conv1d_transpose_1_conv2d_transpose_1_kernel_v_read_readvariableopP
Lsavev2_adam_conv1d_transpose_1_conv2d_transpose_1_bias_v_read_readvariableop>
:savev2_adam_conv_2d_transpose_kernel_v_read_readvariableop<
8savev2_adam_conv_2d_transpose_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_bcd9ca2d57df4c859c2fb0f6035ac936/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*?)
value?)B?)NB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*?
value?B?NB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices? 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop#savev2_z_kernel_read_readvariableop!savev2_z_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableopCsavev2_conv1d_transpose_conv2d_transpose_kernel_read_readvariableopAsavev2_conv1d_transpose_conv2d_transpose_bias_read_readvariableopGsavev2_conv1d_transpose_1_conv2d_transpose_1_kernel_read_readvariableopEsavev2_conv1d_transpose_1_conv2d_transpose_1_bias_read_readvariableop3savev2_conv_2d_transpose_kernel_read_readvariableop1savev2_conv_2d_transpose_bias_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop*savev2_adam_z_kernel_m_read_readvariableop(savev2_adam_z_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableopJsavev2_adam_conv1d_transpose_conv2d_transpose_kernel_m_read_readvariableopHsavev2_adam_conv1d_transpose_conv2d_transpose_bias_m_read_readvariableopNsavev2_adam_conv1d_transpose_1_conv2d_transpose_1_kernel_m_read_readvariableopLsavev2_adam_conv1d_transpose_1_conv2d_transpose_1_bias_m_read_readvariableop:savev2_adam_conv_2d_transpose_kernel_m_read_readvariableop8savev2_adam_conv_2d_transpose_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop*savev2_adam_z_kernel_v_read_readvariableop(savev2_adam_z_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopJsavev2_adam_conv1d_transpose_conv2d_transpose_kernel_v_read_readvariableopHsavev2_adam_conv1d_transpose_conv2d_transpose_bias_v_read_readvariableopNsavev2_adam_conv1d_transpose_1_conv2d_transpose_1_kernel_v_read_readvariableopLsavev2_adam_conv1d_transpose_1_conv2d_transpose_1_bias_v_read_readvariableop:savev2_adam_conv_2d_transpose_kernel_v_read_readvariableop8savev2_adam_conv_2d_transpose_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *\
dtypesR
P2N	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :::::::	?f:f:f::::::f:f:	f?:?:::::::::::::	?f:f:f::::::f:f:	f?:?:::::::::::::	?f:f:f::::::f:f:	f?:?::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 	

_output_shapes
::(
$
"
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?f: 

_output_shapes
:f:$ 

_output_shapes

:f: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:f: 

_output_shapes
:f:%!

_output_shapes
:	f?:!

_output_shapes	
:?:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::( $
"
_output_shapes
:: !

_output_shapes
::("$
"
_output_shapes
:: #

_output_shapes
::%$!

_output_shapes
:	?f: %

_output_shapes
:f:$& 

_output_shapes

:f: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:f: -

_output_shapes
:f:%.!

_output_shapes
:	f?:!/

_output_shapes	
:?:,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
::(8$
"
_output_shapes
:: 9

_output_shapes
::(:$
"
_output_shapes
:: ;

_output_shapes
::%<!

_output_shapes
:	?f: =

_output_shapes
:f:$> 

_output_shapes

:f: ?

_output_shapes
::$@ 

_output_shapes

:: A

_output_shapes
::$B 

_output_shapes

:: C

_output_shapes
::$D 

_output_shapes

:f: E

_output_shapes
:f:%F!

_output_shapes
:	f?:!G

_output_shapes	
:?:,H(
&
_output_shapes
:: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
:: M

_output_shapes
::N

_output_shapes
: 
?
F
*__inference_lambda_6_layer_call_fn_4700093

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_46972592
PartitionedCall}
IdentityIdentityPartitionedCall:output:0*
T0*8
_output_shapes&
$:"??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_4699656

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
D__inference_dense_2_layer_call_and_return_conditional_losses_4696975

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_dense_1_layer_call_and_return_conditional_losses_4696576

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????f:::O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?%
?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4696853

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Elu
IdentityIdentityElu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
a
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_4696376
x
identityc
sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
sub/yZ
subSubxsub/y:output:0*
T0*+
_output_shapes
:?????????d2
subk
	truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
	truediv/yp
truedivRealDivsub:z:0truediv/y:output:0*
T0*+
_output_shapes
:?????????d2	
truedivc
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:N J
+
_output_shapes
:?????????d

_user_specified_namex
?2
?
D__inference_decoder_layer_call_and_return_conditional_losses_4697362
z
dense_2_4697326
dense_2_4697328
dense_3_4697331
dense_3_4697333
dense_4_4697336
dense_4_4697338
conv1d_transpose_4697343
conv1d_transpose_4697345
conv1d_transpose_1_4697348
conv1d_transpose_1_4697350
conv_2d_transpose_4697354
conv_2d_transpose_4697356
identity??(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?)conv_2d_transpose/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallzdense_2_4697326dense_2_4697328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46969752!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4697331dense_3_4697333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46970022!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_4697336dense_4_4697338*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46970292!
dense_4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46970582
reshape/PartitionedCall?
up_sampling1d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling1d_layer_call_and_return_conditional_losses_46968082
up_sampling1d/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall&up_sampling1d/PartitionedCall:output:0conv1d_transpose_4697343conv1d_transpose_4697345*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_46971352*
(conv1d_transpose/StatefulPartitionedCall?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_4697348conv1d_transpose_1_4697350*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_46972282,
*conv1d_transpose_1/StatefulPartitionedCall?
lambda_6/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_46972652
lambda_6/PartitionedCall?
)conv_2d_transpose/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0conv_2d_transpose_4697354conv_2d_transpose_4697356*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv_2d_transpose_layer_call_and_return_conditional_losses_46969502+
)conv_2d_transpose/StatefulPartitionedCall?
lambda_7/PartitionedCallPartitionedCall2conv_2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_46972932
lambda_7/PartitionedCall?
Un_Normalize/PartitionedCallPartitionedCall!lambda_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_46973142
Un_Normalize/PartitionedCall?
IdentityIdentity%Un_Normalize/PartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall*^conv_2d_transpose/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2V
)conv_2d_transpose/StatefulPartitionedCall)conv_2d_transpose/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namez
?
?
D__inference_dense_4_layer_call_and_return_conditional_losses_4699877

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????f:::O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?
?
D__inference_dense_4_layer_call_and_return_conditional_losses_4697029

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????f:::O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?
~
)__inference_dense_3_layer_call_fn_4699866

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46970022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_4696396

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_4700088

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2

ExpandDimsx
IdentityIdentityExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
6__inference_particle_autoencoder_layer_call_fn_4698991
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_46977812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????d

_user_specified_namex
?
?
C__inference_conv1d_layer_call_and_return_conditional_losses_4696475

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2	
BiasAddY
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
Elui
IdentityIdentityElu:activations:0*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????b:::S O
+
_output_shapes
:?????????b
 
_user_specified_nameinputs
?
a
E__inference_lambda_7_layer_call_and_return_conditional_losses_4700108

inputs
identity?
SqueezeSqueezeinputs*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
)__inference_encoder_layer_call_fn_4699259

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_46967682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
4__inference_conv1d_transpose_1_layer_call_fn_4700067

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_46972282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_4697948
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_46963492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1
?
?
B__inference_dense_layer_call_and_return_conditional_losses_4696549

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
D__inference_decoder_layer_call_and_return_conditional_losses_4699416

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resourceN
Jconv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resourceE
Aconv1d_transpose_conv2d_transpose_biasadd_readvariableop_resourceR
Nconv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceI
Econv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource>
:conv_2d_transpose_conv2d_transpose_readvariableop_resource5
1conv_2d_transpose_biasadd_readvariableop_resource
identity??
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddm
dense_2/EluEludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Elu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Elu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
dense_3/BiasAddm
dense_3/EluEludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
dense_3/Elu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Elu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddn
dense_4/EluEludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/Elug
reshape/ShapeShapedense_4/Elu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_4/Elu:activations:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????/2
reshape/Reshapel
up_sampling1d/ConstConst*
_output_shapes
: *
dtype0*
value	B :/2
up_sampling1d/Const?
up_sampling1d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling1d/split/split_dim?	
up_sampling1d/splitSplit&up_sampling1d/split/split_dim:output:0reshape/Reshape:output:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split/2
up_sampling1d/splitx
up_sampling1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling1d/concat/axis?
up_sampling1d/concatConcatV2up_sampling1d/split:output:0up_sampling1d/split:output:0up_sampling1d/split:output:1up_sampling1d/split:output:1up_sampling1d/split:output:2up_sampling1d/split:output:2up_sampling1d/split:output:3up_sampling1d/split:output:3up_sampling1d/split:output:4up_sampling1d/split:output:4up_sampling1d/split:output:5up_sampling1d/split:output:5up_sampling1d/split:output:6up_sampling1d/split:output:6up_sampling1d/split:output:7up_sampling1d/split:output:7up_sampling1d/split:output:8up_sampling1d/split:output:8up_sampling1d/split:output:9up_sampling1d/split:output:9up_sampling1d/split:output:10up_sampling1d/split:output:10up_sampling1d/split:output:11up_sampling1d/split:output:11up_sampling1d/split:output:12up_sampling1d/split:output:12up_sampling1d/split:output:13up_sampling1d/split:output:13up_sampling1d/split:output:14up_sampling1d/split:output:14up_sampling1d/split:output:15up_sampling1d/split:output:15up_sampling1d/split:output:16up_sampling1d/split:output:16up_sampling1d/split:output:17up_sampling1d/split:output:17up_sampling1d/split:output:18up_sampling1d/split:output:18up_sampling1d/split:output:19up_sampling1d/split:output:19up_sampling1d/split:output:20up_sampling1d/split:output:20up_sampling1d/split:output:21up_sampling1d/split:output:21up_sampling1d/split:output:22up_sampling1d/split:output:22up_sampling1d/split:output:23up_sampling1d/split:output:23up_sampling1d/split:output:24up_sampling1d/split:output:24up_sampling1d/split:output:25up_sampling1d/split:output:25up_sampling1d/split:output:26up_sampling1d/split:output:26up_sampling1d/split:output:27up_sampling1d/split:output:27up_sampling1d/split:output:28up_sampling1d/split:output:28up_sampling1d/split:output:29up_sampling1d/split:output:29up_sampling1d/split:output:30up_sampling1d/split:output:30up_sampling1d/split:output:31up_sampling1d/split:output:31up_sampling1d/split:output:32up_sampling1d/split:output:32up_sampling1d/split:output:33up_sampling1d/split:output:33up_sampling1d/split:output:34up_sampling1d/split:output:34up_sampling1d/split:output:35up_sampling1d/split:output:35up_sampling1d/split:output:36up_sampling1d/split:output:36up_sampling1d/split:output:37up_sampling1d/split:output:37up_sampling1d/split:output:38up_sampling1d/split:output:38up_sampling1d/split:output:39up_sampling1d/split:output:39up_sampling1d/split:output:40up_sampling1d/split:output:40up_sampling1d/split:output:41up_sampling1d/split:output:41up_sampling1d/split:output:42up_sampling1d/split:output:42up_sampling1d/split:output:43up_sampling1d/split:output:43up_sampling1d/split:output:44up_sampling1d/split:output:44up_sampling1d/split:output:45up_sampling1d/split:output:45up_sampling1d/split:output:46up_sampling1d/split:output:46"up_sampling1d/concat/axis:output:0*
N^*
T0*+
_output_shapes
:?????????^2
up_sampling1d/concat?
(conv1d_transpose/lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(conv1d_transpose/lambda_2/ExpandDims/dim?
$conv1d_transpose/lambda_2/ExpandDims
ExpandDimsup_sampling1d/concat:output:01conv1d_transpose/lambda_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2&
$conv1d_transpose/lambda_2/ExpandDims?
'conv1d_transpose/conv2d_transpose/ShapeShape-conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*
_output_shapes
:2)
'conv1d_transpose/conv2d_transpose/Shape?
5conv1d_transpose/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5conv1d_transpose/conv2d_transpose/strided_slice/stack?
7conv1d_transpose/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv2d_transpose/strided_slice/stack_1?
7conv1d_transpose/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv2d_transpose/strided_slice/stack_2?
/conv1d_transpose/conv2d_transpose/strided_sliceStridedSlice0conv1d_transpose/conv2d_transpose/Shape:output:0>conv1d_transpose/conv2d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv2d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/conv1d_transpose/conv2d_transpose/strided_slice?
)conv1d_transpose/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2+
)conv1d_transpose/conv2d_transpose/stack/1?
)conv1d_transpose/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)conv1d_transpose/conv2d_transpose/stack/2?
)conv1d_transpose/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)conv1d_transpose/conv2d_transpose/stack/3?
'conv1d_transpose/conv2d_transpose/stackPack8conv1d_transpose/conv2d_transpose/strided_slice:output:02conv1d_transpose/conv2d_transpose/stack/1:output:02conv1d_transpose/conv2d_transpose/stack/2:output:02conv1d_transpose/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'conv1d_transpose/conv2d_transpose/stack?
7conv1d_transpose/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose/conv2d_transpose/strided_slice_1/stack?
9conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1?
9conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2?
1conv1d_transpose/conv2d_transpose/strided_slice_1StridedSlice0conv1d_transpose/conv2d_transpose/stack:output:0@conv1d_transpose/conv2d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv2d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1conv1d_transpose/conv2d_transpose/strided_slice_1?
Aconv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpJconv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02C
Aconv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp?
2conv1d_transpose/conv2d_transpose/conv2d_transposeConv2DBackpropInput0conv1d_transpose/conv2d_transpose/stack:output:0Iconv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0-conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
24
2conv1d_transpose/conv2d_transpose/conv2d_transpose?
8conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpAconv1d_transpose_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp?
)conv1d_transpose/conv2d_transpose/BiasAddBiasAdd;conv1d_transpose/conv2d_transpose/conv2d_transpose:output:0@conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`2+
)conv1d_transpose/conv2d_transpose/BiasAdd?
%conv1d_transpose/conv2d_transpose/EluElu2conv1d_transpose/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`2'
%conv1d_transpose/conv2d_transpose/Elu?
!conv1d_transpose/lambda_3/SqueezeSqueeze3conv1d_transpose/conv2d_transpose/Elu:activations:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims
2#
!conv1d_transpose/lambda_3/Squeeze?
*conv1d_transpose_1/lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*conv1d_transpose_1/lambda_4/ExpandDims/dim?
&conv1d_transpose_1/lambda_4/ExpandDims
ExpandDims*conv1d_transpose/lambda_3/Squeeze:output:03conv1d_transpose_1/lambda_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2(
&conv1d_transpose_1/lambda_4/ExpandDims?
+conv1d_transpose_1/conv2d_transpose_1/ShapeShape/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*
_output_shapes
:2-
+conv1d_transpose_1/conv2d_transpose_1/Shape?
9conv1d_transpose_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack?
;conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1?
;conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2?
3conv1d_transpose_1/conv2d_transpose_1/strided_sliceStridedSlice4conv1d_transpose_1/conv2d_transpose_1/Shape:output:0Bconv1d_transpose_1/conv2d_transpose_1/strided_slice/stack:output:0Dconv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1:output:0Dconv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3conv1d_transpose_1/conv2d_transpose_1/strided_slice?
-conv1d_transpose_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b2/
-conv1d_transpose_1/conv2d_transpose_1/stack/1?
-conv1d_transpose_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-conv1d_transpose_1/conv2d_transpose_1/stack/2?
-conv1d_transpose_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-conv1d_transpose_1/conv2d_transpose_1/stack/3?
+conv1d_transpose_1/conv2d_transpose_1/stackPack<conv1d_transpose_1/conv2d_transpose_1/strided_slice:output:06conv1d_transpose_1/conv2d_transpose_1/stack/1:output:06conv1d_transpose_1/conv2d_transpose_1/stack/2:output:06conv1d_transpose_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+conv1d_transpose_1/conv2d_transpose_1/stack?
;conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack?
=conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1?
=conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2?
5conv1d_transpose_1/conv2d_transpose_1/strided_slice_1StridedSlice4conv1d_transpose_1/conv2d_transpose_1/stack:output:0Dconv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack:output:0Fconv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Fconv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5conv1d_transpose_1/conv2d_transpose_1/strided_slice_1?
Econv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpNconv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02G
Econv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
6conv1d_transpose_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput4conv1d_transpose_1/conv2d_transpose_1/stack:output:0Mconv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
28
6conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose?
<conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpEconv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
-conv1d_transpose_1/conv2d_transpose_1/BiasAddBiasAdd?conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose:output:0Dconv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2/
-conv1d_transpose_1/conv2d_transpose_1/BiasAdd?
)conv1d_transpose_1/conv2d_transpose_1/EluElu6conv1d_transpose_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2+
)conv1d_transpose_1/conv2d_transpose_1/Elu?
#conv1d_transpose_1/lambda_5/SqueezeSqueeze7conv1d_transpose_1/conv2d_transpose_1/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2%
#conv1d_transpose_1/lambda_5/Squeezet
lambda_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_6/ExpandDims/dim?
lambda_6/ExpandDims
ExpandDims,conv1d_transpose_1/lambda_5/Squeeze:output:0 lambda_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
lambda_6/ExpandDims~
conv_2d_transpose/ShapeShapelambda_6/ExpandDims:output:0*
T0*
_output_shapes
:2
conv_2d_transpose/Shape?
%conv_2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%conv_2d_transpose/strided_slice/stack?
'conv_2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'conv_2d_transpose/strided_slice/stack_1?
'conv_2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'conv_2d_transpose/strided_slice/stack_2?
conv_2d_transpose/strided_sliceStridedSlice conv_2d_transpose/Shape:output:0.conv_2d_transpose/strided_slice/stack:output:00conv_2d_transpose/strided_slice/stack_1:output:00conv_2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
conv_2d_transpose/strided_slicex
conv_2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d2
conv_2d_transpose/stack/1x
conv_2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_2d_transpose/stack/2x
conv_2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_2d_transpose/stack/3?
conv_2d_transpose/stackPack(conv_2d_transpose/strided_slice:output:0"conv_2d_transpose/stack/1:output:0"conv_2d_transpose/stack/2:output:0"conv_2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_2d_transpose/stack?
'conv_2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv_2d_transpose/strided_slice_1/stack?
)conv_2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv_2d_transpose/strided_slice_1/stack_1?
)conv_2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv_2d_transpose/strided_slice_1/stack_2?
!conv_2d_transpose/strided_slice_1StridedSlice conv_2d_transpose/stack:output:00conv_2d_transpose/strided_slice_1/stack:output:02conv_2d_transpose/strided_slice_1/stack_1:output:02conv_2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv_2d_transpose/strided_slice_1?
1conv_2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp:conv_2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv_2d_transpose/conv2d_transpose/ReadVariableOp?
"conv_2d_transpose/conv2d_transposeConv2DBackpropInput conv_2d_transpose/stack:output:09conv_2d_transpose/conv2d_transpose/ReadVariableOp:value:0lambda_6/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2$
"conv_2d_transpose/conv2d_transpose?
(conv_2d_transpose/BiasAdd/ReadVariableOpReadVariableOp1conv_2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(conv_2d_transpose/BiasAdd/ReadVariableOp?
conv_2d_transpose/BiasAddBiasAdd+conv_2d_transpose/conv2d_transpose:output:00conv_2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2
conv_2d_transpose/BiasAdd?
lambda_7/SqueezeSqueeze"conv_2d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
2
lambda_7/Squeeze}
Un_Normalize/mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
Un_Normalize/mul/y?
Un_Normalize/mulMullambda_7/Squeeze:output:0Un_Normalize/mul/y:output:0*
T0*+
_output_shapes
:?????????d2
Un_Normalize/mul}
Un_Normalize/add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
Un_Normalize/add/y?
Un_Normalize/addAddV2Un_Normalize/mul:z:0Un_Normalize/add/y:output:0*
T0*+
_output_shapes
:?????????d2
Un_Normalize/addl
IdentityIdentityUn_Normalize/add:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_1_layer_call_and_return_conditional_losses_4696507

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2	
BiasAddY
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:?????????^2
Elui
IdentityIdentityElu:activations:0*
T0*+
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????`:::S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
F
/__inference_Std_Normalize_layer_call_fn_4699644
x
identity?
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_46963762
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:N J
+
_output_shapes
:?????????d

_user_specified_namex
?
`
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_4697314
x
identityc
mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
mul/yc
mulMulxmul/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
mulc
add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
add/yk
addAddV2mul:z:0add/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
addh
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:` \
=
_output_shapes+
):'???????????????????????????

_user_specified_namex
?
?
D__inference_dense_3_layer_call_and_return_conditional_losses_4699857

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_4696390

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
E__inference_lambda_7_layer_call_and_return_conditional_losses_4697288

inputs
identity?
SqueezeSqueezeinputs*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
>__inference_z_layer_call_and_return_conditional_losses_4699817

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_4697058

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????/2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????/2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_2_layer_call_fn_4699846

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46969752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ɨ
?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698717
x1
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource>
:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource2
.encoder_conv1d_biasadd_readvariableop_resource@
<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_1_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.encoder_dense_1_matmul_readvariableop_resource3
/encoder_dense_1_biasadd_readvariableop_resource,
(encoder_z_matmul_readvariableop_resource-
)encoder_z_biasadd_readvariableop_resource2
.decoder_dense_2_matmul_readvariableop_resource3
/decoder_dense_2_biasadd_readvariableop_resource2
.decoder_dense_3_matmul_readvariableop_resource3
/decoder_dense_3_biasadd_readvariableop_resource2
.decoder_dense_4_matmul_readvariableop_resource3
/decoder_dense_4_biasadd_readvariableop_resourceV
Rdecoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resourceM
Idecoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resourceZ
Vdecoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceQ
Mdecoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resourceF
Bdecoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource=
9decoder_conv_2d_transpose_biasadd_readvariableop_resource
identity??
encoder/Std_Normalize/sub/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
encoder/Std_Normalize/sub/y?
encoder/Std_Normalize/subSubx$encoder/Std_Normalize/sub/y:output:0*
T0*+
_output_shapes
:?????????d2
encoder/Std_Normalize/sub?
encoder/Std_Normalize/truediv/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2!
encoder/Std_Normalize/truediv/y?
encoder/Std_Normalize/truedivRealDivencoder/Std_Normalize/sub:z:0(encoder/Std_Normalize/truediv/y:output:0*
T0*+
_output_shapes
:?????????d2
encoder/Std_Normalize/truediv?
encoder/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
encoder/lambda/ExpandDims/dim?
encoder/lambda/ExpandDims
ExpandDims!encoder/Std_Normalize/truediv:z:0&encoder/lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d2
encoder/lambda/ExpandDims?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2D"encoder/lambda/ExpandDims:output:0,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2
encoder/conv2d/BiasAdd?
encoder/conv2d/EluEluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b2
encoder/conv2d/Elu?
encoder/lambda_1/SqueezeSqueeze encoder/conv2d/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2
encoder/lambda_1/Squeeze?
$encoder/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$encoder/conv1d/conv1d/ExpandDims/dim?
 encoder/conv1d/conv1d/ExpandDims
ExpandDims!encoder/lambda_1/Squeeze:output:0-encoder/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2"
 encoder/conv1d/conv1d/ExpandDims?
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype023
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
&encoder/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&encoder/conv1d/conv1d/ExpandDims_1/dim?
"encoder/conv1d/conv1d/ExpandDims_1
ExpandDims9encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0/encoder/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2$
"encoder/conv1d/conv1d/ExpandDims_1?
encoder/conv1d/conv1dConv2D)encoder/conv1d/conv1d/ExpandDims:output:0+encoder/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2
encoder/conv1d/conv1d?
encoder/conv1d/conv1d/SqueezeSqueezeencoder/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????2
encoder/conv1d/conv1d/Squeeze?
%encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv1d/BiasAdd/ReadVariableOp?
encoder/conv1d/BiasAddBiasAdd&encoder/conv1d/conv1d/Squeeze:output:0-encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
encoder/conv1d/BiasAdd?
encoder/conv1d/EluEluencoder/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
encoder/conv1d/Elu?
&encoder/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&encoder/conv1d_1/conv1d/ExpandDims/dim?
"encoder/conv1d_1/conv1d/ExpandDims
ExpandDims encoder/conv1d/Elu:activations:0/encoder/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`2$
"encoder/conv1d_1/conv1d/ExpandDims?
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype025
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
(encoder/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_1/conv1d/ExpandDims_1/dim?
$encoder/conv1d_1/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2&
$encoder/conv1d_1/conv1d/ExpandDims_1?
encoder/conv1d_1/conv1dConv2D+encoder/conv1d_1/conv1d/ExpandDims:output:0-encoder/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^*
paddingVALID*
strides
2
encoder/conv1d_1/conv1d?
encoder/conv1d_1/conv1d/SqueezeSqueeze encoder/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????^*
squeeze_dims

?????????2!
encoder/conv1d_1/conv1d/Squeeze?
'encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'encoder/conv1d_1/BiasAdd/ReadVariableOp?
encoder/conv1d_1/BiasAddBiasAdd(encoder/conv1d_1/conv1d/Squeeze:output:0/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^2
encoder/conv1d_1/BiasAdd?
encoder/conv1d_1/EluElu!encoder/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^2
encoder/conv1d_1/Elu?
(encoder/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(encoder/average_pooling1d/ExpandDims/dim?
$encoder/average_pooling1d/ExpandDims
ExpandDims"encoder/conv1d_1/Elu:activations:01encoder/average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2&
$encoder/average_pooling1d/ExpandDims?
!encoder/average_pooling1d/AvgPoolAvgPool-encoder/average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2#
!encoder/average_pooling1d/AvgPool?
!encoder/average_pooling1d/SqueezeSqueeze*encoder/average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2#
!encoder/average_pooling1d/Squeeze
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape*encoder/average_pooling1d/Squeeze:output:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?f*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/BiasAdd
encoder/dense/EluEluencoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
encoder/dense/Elu?
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02'
%encoder/dense_1/MatMul/ReadVariableOp?
encoder/dense_1/MatMulMatMulencoder/dense/Elu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/MatMul?
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_1/BiasAdd/ReadVariableOp?
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/BiasAdd?
encoder/dense_1/EluElu encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/Elu?
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
encoder/z/MatMul/ReadVariableOp?
encoder/z/MatMulMatMul!encoder/dense_1/Elu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z/MatMul?
 encoder/z/BiasAdd/ReadVariableOpReadVariableOp)encoder_z_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 encoder/z/BiasAdd/ReadVariableOp?
encoder/z/BiasAddBiasAddencoder/z/MatMul:product:0(encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z/BiasAdd?
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%decoder/dense_2/MatMul/ReadVariableOp?
decoder/dense_2/MatMulMatMulencoder/z/BiasAdd:output:0-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/MatMul?
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&decoder/dense_2/BiasAdd/ReadVariableOp?
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/BiasAdd?
decoder/dense_2/EluElu decoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
decoder/dense_2/Elu?
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource*
_output_shapes

:f*
dtype02'
%decoder/dense_3/MatMul/ReadVariableOp?
decoder/dense_3/MatMulMatMul!decoder/dense_2/Elu:activations:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/MatMul?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype02(
&decoder/dense_3/BiasAdd/ReadVariableOp?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/BiasAdd?
decoder/dense_3/EluElu decoder/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f2
decoder/dense_3/Elu?
%decoder/dense_4/MatMul/ReadVariableOpReadVariableOp.decoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	f?*
dtype02'
%decoder/dense_4/MatMul/ReadVariableOp?
decoder/dense_4/MatMulMatMul!decoder/dense_3/Elu:activations:0-decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/MatMul?
&decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_4/BiasAdd/ReadVariableOp?
decoder/dense_4/BiasAddBiasAdd decoder/dense_4/MatMul:product:0.decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/BiasAdd?
decoder/dense_4/EluElu decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
decoder/dense_4/Elu
decoder/reshape/ShapeShape!decoder/dense_4/Elu:activations:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :/2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshape!decoder/dense_4/Elu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????/2
decoder/reshape/Reshape|
decoder/up_sampling1d/ConstConst*
_output_shapes
: *
dtype0*
value	B :/2
decoder/up_sampling1d/Const?
%decoder/up_sampling1d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%decoder/up_sampling1d/split/split_dim?

decoder/up_sampling1d/splitSplit.decoder/up_sampling1d/split/split_dim:output:0 decoder/reshape/Reshape:output:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split/2
decoder/up_sampling1d/split?
!decoder/up_sampling1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/up_sampling1d/concat/axis?
decoder/up_sampling1d/concatConcatV2$decoder/up_sampling1d/split:output:0$decoder/up_sampling1d/split:output:0$decoder/up_sampling1d/split:output:1$decoder/up_sampling1d/split:output:1$decoder/up_sampling1d/split:output:2$decoder/up_sampling1d/split:output:2$decoder/up_sampling1d/split:output:3$decoder/up_sampling1d/split:output:3$decoder/up_sampling1d/split:output:4$decoder/up_sampling1d/split:output:4$decoder/up_sampling1d/split:output:5$decoder/up_sampling1d/split:output:5$decoder/up_sampling1d/split:output:6$decoder/up_sampling1d/split:output:6$decoder/up_sampling1d/split:output:7$decoder/up_sampling1d/split:output:7$decoder/up_sampling1d/split:output:8$decoder/up_sampling1d/split:output:8$decoder/up_sampling1d/split:output:9$decoder/up_sampling1d/split:output:9%decoder/up_sampling1d/split:output:10%decoder/up_sampling1d/split:output:10%decoder/up_sampling1d/split:output:11%decoder/up_sampling1d/split:output:11%decoder/up_sampling1d/split:output:12%decoder/up_sampling1d/split:output:12%decoder/up_sampling1d/split:output:13%decoder/up_sampling1d/split:output:13%decoder/up_sampling1d/split:output:14%decoder/up_sampling1d/split:output:14%decoder/up_sampling1d/split:output:15%decoder/up_sampling1d/split:output:15%decoder/up_sampling1d/split:output:16%decoder/up_sampling1d/split:output:16%decoder/up_sampling1d/split:output:17%decoder/up_sampling1d/split:output:17%decoder/up_sampling1d/split:output:18%decoder/up_sampling1d/split:output:18%decoder/up_sampling1d/split:output:19%decoder/up_sampling1d/split:output:19%decoder/up_sampling1d/split:output:20%decoder/up_sampling1d/split:output:20%decoder/up_sampling1d/split:output:21%decoder/up_sampling1d/split:output:21%decoder/up_sampling1d/split:output:22%decoder/up_sampling1d/split:output:22%decoder/up_sampling1d/split:output:23%decoder/up_sampling1d/split:output:23%decoder/up_sampling1d/split:output:24%decoder/up_sampling1d/split:output:24%decoder/up_sampling1d/split:output:25%decoder/up_sampling1d/split:output:25%decoder/up_sampling1d/split:output:26%decoder/up_sampling1d/split:output:26%decoder/up_sampling1d/split:output:27%decoder/up_sampling1d/split:output:27%decoder/up_sampling1d/split:output:28%decoder/up_sampling1d/split:output:28%decoder/up_sampling1d/split:output:29%decoder/up_sampling1d/split:output:29%decoder/up_sampling1d/split:output:30%decoder/up_sampling1d/split:output:30%decoder/up_sampling1d/split:output:31%decoder/up_sampling1d/split:output:31%decoder/up_sampling1d/split:output:32%decoder/up_sampling1d/split:output:32%decoder/up_sampling1d/split:output:33%decoder/up_sampling1d/split:output:33%decoder/up_sampling1d/split:output:34%decoder/up_sampling1d/split:output:34%decoder/up_sampling1d/split:output:35%decoder/up_sampling1d/split:output:35%decoder/up_sampling1d/split:output:36%decoder/up_sampling1d/split:output:36%decoder/up_sampling1d/split:output:37%decoder/up_sampling1d/split:output:37%decoder/up_sampling1d/split:output:38%decoder/up_sampling1d/split:output:38%decoder/up_sampling1d/split:output:39%decoder/up_sampling1d/split:output:39%decoder/up_sampling1d/split:output:40%decoder/up_sampling1d/split:output:40%decoder/up_sampling1d/split:output:41%decoder/up_sampling1d/split:output:41%decoder/up_sampling1d/split:output:42%decoder/up_sampling1d/split:output:42%decoder/up_sampling1d/split:output:43%decoder/up_sampling1d/split:output:43%decoder/up_sampling1d/split:output:44%decoder/up_sampling1d/split:output:44%decoder/up_sampling1d/split:output:45%decoder/up_sampling1d/split:output:45%decoder/up_sampling1d/split:output:46%decoder/up_sampling1d/split:output:46*decoder/up_sampling1d/concat/axis:output:0*
N^*
T0*+
_output_shapes
:?????????^2
decoder/up_sampling1d/concat?
0decoder/conv1d_transpose/lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0decoder/conv1d_transpose/lambda_2/ExpandDims/dim?
,decoder/conv1d_transpose/lambda_2/ExpandDims
ExpandDims%decoder/up_sampling1d/concat:output:09decoder/conv1d_transpose/lambda_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^2.
,decoder/conv1d_transpose/lambda_2/ExpandDims?
/decoder/conv1d_transpose/conv2d_transpose/ShapeShape5decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*
_output_shapes
:21
/decoder/conv1d_transpose/conv2d_transpose/Shape?
=decoder/conv1d_transpose/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2?
7decoder/conv1d_transpose/conv2d_transpose/strided_sliceStridedSlice8decoder/conv1d_transpose/conv2d_transpose/Shape:output:0Fdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_1:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7decoder/conv1d_transpose/conv2d_transpose/strided_slice?
1decoder/conv1d_transpose/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`23
1decoder/conv1d_transpose/conv2d_transpose/stack/1?
1decoder/conv1d_transpose/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :23
1decoder/conv1d_transpose/conv2d_transpose/stack/2?
1decoder/conv1d_transpose/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :23
1decoder/conv1d_transpose/conv2d_transpose/stack/3?
/decoder/conv1d_transpose/conv2d_transpose/stackPack@decoder/conv1d_transpose/conv2d_transpose/strided_slice:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/1:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/2:output:0:decoder/conv1d_transpose/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:21
/decoder/conv1d_transpose/conv2d_transpose/stack?
?decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?decoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack?
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1?
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Adecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2?
9decoder/conv1d_transpose/conv2d_transpose/strided_slice_1StridedSlice8decoder/conv1d_transpose/conv2d_transpose/stack:output:0Hdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack:output:0Jdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_1:output:0Jdecoder/conv1d_transpose/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9decoder/conv1d_transpose/conv2d_transpose/strided_slice_1?
Idecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpRdecoder_conv1d_transpose_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02K
Idecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp?
:decoder/conv1d_transpose/conv2d_transpose/conv2d_transposeConv2DBackpropInput8decoder/conv1d_transpose/conv2d_transpose/stack:output:0Qdecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:05decoder/conv1d_transpose/lambda_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2<
:decoder/conv1d_transpose/conv2d_transpose/conv2d_transpose?
@decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpIdecoder_conv1d_transpose_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@decoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp?
1decoder/conv1d_transpose/conv2d_transpose/BiasAddBiasAddCdecoder/conv1d_transpose/conv2d_transpose/conv2d_transpose:output:0Hdecoder/conv1d_transpose/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`23
1decoder/conv1d_transpose/conv2d_transpose/BiasAdd?
-decoder/conv1d_transpose/conv2d_transpose/EluElu:decoder/conv1d_transpose/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`2/
-decoder/conv1d_transpose/conv2d_transpose/Elu?
)decoder/conv1d_transpose/lambda_3/SqueezeSqueeze;decoder/conv1d_transpose/conv2d_transpose/Elu:activations:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims
2+
)decoder/conv1d_transpose/lambda_3/Squeeze?
2decoder/conv1d_transpose_1/lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim?
.decoder/conv1d_transpose_1/lambda_4/ExpandDims
ExpandDims2decoder/conv1d_transpose/lambda_3/Squeeze:output:0;decoder/conv1d_transpose_1/lambda_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????`20
.decoder/conv1d_transpose_1/lambda_4/ExpandDims?
3decoder/conv1d_transpose_1/conv2d_transpose_1/ShapeShape7decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*
_output_shapes
:25
3decoder/conv1d_transpose_1/conv2d_transpose_1/Shape?
Adecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Adecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2?
;decoder/conv1d_transpose_1/conv2d_transpose_1/strided_sliceStridedSlice<decoder/conv1d_transpose_1/conv2d_transpose_1/Shape:output:0Jdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_1:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2?
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :27
5decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3?
3decoder/conv1d_transpose_1/conv2d_transpose_1/stackPackDdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/1:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/2:output:0>decoder/conv1d_transpose_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:25
3decoder/conv1d_transpose_1/conv2d_transpose_1/stack?
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cdecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack?
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1?
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Edecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2?
=decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1StridedSlice<decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack:output:0Ndecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ndecoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=decoder/conv1d_transpose_1/conv2d_transpose_1/strided_slice_1?
Mdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpVdecoder_conv1d_transpose_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02O
Mdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
>decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput<decoder/conv1d_transpose_1/conv2d_transpose_1/stack:output:0Udecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:07decoder/conv1d_transpose_1/lambda_4/ExpandDims:output:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2@
>decoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose?
Ddecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpMdecoder_conv1d_transpose_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02F
Ddecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
5decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAddBiasAddGdecoder/conv1d_transpose_1/conv2d_transpose_1/conv2d_transpose:output:0Ldecoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b27
5decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd?
1decoder/conv1d_transpose_1/conv2d_transpose_1/EluElu>decoder/conv1d_transpose_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????b23
1decoder/conv1d_transpose_1/conv2d_transpose_1/Elu?
+decoder/conv1d_transpose_1/lambda_5/SqueezeSqueeze?decoder/conv1d_transpose_1/conv2d_transpose_1/Elu:activations:0*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2-
+decoder/conv1d_transpose_1/lambda_5/Squeeze?
decoder/lambda_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
decoder/lambda_6/ExpandDims/dim?
decoder/lambda_6/ExpandDims
ExpandDims4decoder/conv1d_transpose_1/lambda_5/Squeeze:output:0(decoder/lambda_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
decoder/lambda_6/ExpandDims?
decoder/conv_2d_transpose/ShapeShape$decoder/lambda_6/ExpandDims:output:0*
T0*
_output_shapes
:2!
decoder/conv_2d_transpose/Shape?
-decoder/conv_2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-decoder/conv_2d_transpose/strided_slice/stack?
/decoder/conv_2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/decoder/conv_2d_transpose/strided_slice/stack_1?
/decoder/conv_2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/decoder/conv_2d_transpose/strided_slice/stack_2?
'decoder/conv_2d_transpose/strided_sliceStridedSlice(decoder/conv_2d_transpose/Shape:output:06decoder/conv_2d_transpose/strided_slice/stack:output:08decoder/conv_2d_transpose/strided_slice/stack_1:output:08decoder/conv_2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'decoder/conv_2d_transpose/strided_slice?
!decoder/conv_2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d2#
!decoder/conv_2d_transpose/stack/1?
!decoder/conv_2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/conv_2d_transpose/stack/2?
!decoder/conv_2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/conv_2d_transpose/stack/3?
decoder/conv_2d_transpose/stackPack0decoder/conv_2d_transpose/strided_slice:output:0*decoder/conv_2d_transpose/stack/1:output:0*decoder/conv_2d_transpose/stack/2:output:0*decoder/conv_2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/conv_2d_transpose/stack?
/decoder/conv_2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv_2d_transpose/strided_slice_1/stack?
1decoder/conv_2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv_2d_transpose/strided_slice_1/stack_1?
1decoder/conv_2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv_2d_transpose/strided_slice_1/stack_2?
)decoder/conv_2d_transpose/strided_slice_1StridedSlice(decoder/conv_2d_transpose/stack:output:08decoder/conv_2d_transpose/strided_slice_1/stack:output:0:decoder/conv_2d_transpose/strided_slice_1/stack_1:output:0:decoder/conv_2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv_2d_transpose/strided_slice_1?
9decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpBdecoder_conv_2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02;
9decoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp?
*decoder/conv_2d_transpose/conv2d_transposeConv2DBackpropInput(decoder/conv_2d_transpose/stack:output:0Adecoder/conv_2d_transpose/conv2d_transpose/ReadVariableOp:value:0$decoder/lambda_6/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2,
*decoder/conv_2d_transpose/conv2d_transpose?
0decoder/conv_2d_transpose/BiasAdd/ReadVariableOpReadVariableOp9decoder_conv_2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0decoder/conv_2d_transpose/BiasAdd/ReadVariableOp?
!decoder/conv_2d_transpose/BiasAddBiasAdd3decoder/conv_2d_transpose/conv2d_transpose:output:08decoder/conv_2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2#
!decoder/conv_2d_transpose/BiasAdd?
decoder/lambda_7/SqueezeSqueeze*decoder/conv_2d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
2
decoder/lambda_7/Squeeze?
decoder/Un_Normalize/mul/yConst*
_output_shapes
:*
dtype0*!
valueB"հL>STQ>?-?A2
decoder/Un_Normalize/mul/y?
decoder/Un_Normalize/mulMul!decoder/lambda_7/Squeeze:output:0#decoder/Un_Normalize/mul/y:output:0*
T0*+
_output_shapes
:?????????d2
decoder/Un_Normalize/mul?
decoder/Un_Normalize/add/yConst*
_output_shapes
:*
dtype0*!
valueB"?]?9????<?Y@2
decoder/Un_Normalize/add/y?
decoder/Un_Normalize/addAddV2decoder/Un_Normalize/mul:z:0#decoder/Un_Normalize/add/y:output:0*
T0*+
_output_shapes
:?????????d2
decoder/Un_Normalize/addt
IdentityIdentitydecoder/Un_Normalize/add:z:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d:::::::::::::::::::::::::N J
+
_output_shapes
:?????????d

_user_specified_namex
?
?
2__inference_conv1d_transpose_layer_call_fn_4699981

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_46971352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?,
?
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_4697135

inputs=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource
identity?t
lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda_2/ExpandDims/dim?
lambda_2/ExpandDims
ExpandDimsinputs lambda_2/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
lambda_2/ExpandDims|
conv2d_transpose/ShapeShapelambda_2/ExpandDims:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulr
conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/add/y?
conv2d_transpose/addAddV2conv2d_transpose/mul:z:0conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/addv
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/add:z:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_2/stack?
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1?
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0lambda_2/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose/BiasAdd?
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv2d_transpose/Elu?
lambda_3/SqueezeSqueeze"conv2d_transpose/Elu:activations:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
lambda_3/Squeezez
IdentityIdentitylambda_3/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'???????????????????????????:::e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
__inference_threeD_loss_4699065

inputs
outputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimy

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*'
_output_shapes
:?d2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsoutputsExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?d2
ExpandDims_1?
SquaredDifferenceSquaredDifferenceExpandDims:output:0ExpandDims_1:output:0*
T0*'
_output_shapes
:?dd2
SquaredDifferencey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indicesv
SumSumSquaredDifference:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?dd2
Sump
Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Min/reduction_indicesi
MinMinSum:output:0Min/reduction_indices:output:0*
T0*
_output_shapes
:	?d2
Mint
Min_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Min_1/reduction_indiceso
Min_1MinSum:output:0 Min_1/reduction_indices:output:0*
T0*
_output_shapes
:	?d2
Min_1r
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesi
MeanMeanMin:output:0Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2
Meanv
Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean_1/reduction_indicesq
Mean_1MeanMin_1:output:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes	
:?2
Mean_1Y
addAddV2Mean:output:0Mean_1:output:0*
T0*
_output_shapes	
:?2
addO
IdentityIdentityadd:z:0*
T0*
_output_shapes	
:?2

Identity"
identityIdentity:output:0*1
_input_shapes 
:?d:?d:K G
#
_output_shapes
:?d
 
_user_specified_nameinputs:LH
#
_output_shapes
:?d
!
_user_specified_name	outputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_4696530

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_4697259

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2

ExpandDimsx
IdentityIdentityExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?2
?
D__inference_decoder_layer_call_and_return_conditional_losses_4697323
z
dense_2_4696986
dense_2_4696988
dense_3_4697013
dense_3_4697015
dense_4_4697040
dense_4_4697042
conv1d_transpose_4697155
conv1d_transpose_4697157
conv1d_transpose_1_4697248
conv1d_transpose_1_4697250
conv_2d_transpose_4697278
conv_2d_transpose_4697280
identity??(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?)conv_2d_transpose/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallzdense_2_4696986dense_2_4696988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46969752!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4697013dense_3_4697015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46970022!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_4697040dense_4_4697042*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46970292!
dense_4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46970582
reshape/PartitionedCall?
up_sampling1d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling1d_layer_call_and_return_conditional_losses_46968082
up_sampling1d/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall&up_sampling1d/PartitionedCall:output:0conv1d_transpose_4697155conv1d_transpose_4697157*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_46971012*
(conv1d_transpose/StatefulPartitionedCall?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_4697248conv1d_transpose_1_4697250*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_46971942,
*conv1d_transpose_1/StatefulPartitionedCall?
lambda_6/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_46972592
lambda_6/PartitionedCall?
)conv_2d_transpose/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0conv_2d_transpose_4697278conv_2d_transpose_4697280*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv_2d_transpose_layer_call_and_return_conditional_losses_46969502+
)conv_2d_transpose/StatefulPartitionedCall?
lambda_7/PartitionedCallPartitionedCall2conv_2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_46972882
lambda_7/PartitionedCall?
Un_Normalize/PartitionedCallPartitionedCall!lambda_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_46973142
Un_Normalize/PartitionedCall?
IdentityIdentity%Un_Normalize/PartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall*^conv_2d_transpose/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2V
)conv_2d_transpose/StatefulPartitionedCall)conv_2d_transpose/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namez
?
E
)__inference_reshape_layer_call_fn_4699904

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46970582
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????/2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_layer_call_and_return_conditional_losses_4699677

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????b2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:?????????b2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d:::W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_4696446

inputs
identityr
SqueezeSqueezeinputs*
T0*+
_output_shapes
:?????????b*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????b2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????b:W S
/
_output_shapes
:?????????b
 
_user_specified_nameinputs
?2
?
D__inference_decoder_layer_call_and_return_conditional_losses_4697472

inputs
dense_2_4697436
dense_2_4697438
dense_3_4697441
dense_3_4697443
dense_4_4697446
dense_4_4697448
conv1d_transpose_4697453
conv1d_transpose_4697455
conv1d_transpose_1_4697458
conv1d_transpose_1_4697460
conv_2d_transpose_4697464
conv_2d_transpose_4697466
identity??(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?)conv_2d_transpose/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_4697436dense_2_4697438*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46969752!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4697441dense_3_4697443*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46970022!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_4697446dense_4_4697448*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46970292!
dense_4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46970582
reshape/PartitionedCall?
up_sampling1d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling1d_layer_call_and_return_conditional_losses_46968082
up_sampling1d/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall&up_sampling1d/PartitionedCall:output:0conv1d_transpose_4697453conv1d_transpose_4697455*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_46971352*
(conv1d_transpose/StatefulPartitionedCall?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_4697458conv1d_transpose_1_4697460*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_46972282,
*conv1d_transpose_1/StatefulPartitionedCall?
lambda_6/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_46972652
lambda_6/PartitionedCall?
)conv_2d_transpose/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0conv_2d_transpose_4697464conv_2d_transpose_4697466*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv_2d_transpose_layer_call_and_return_conditional_losses_46969502+
)conv_2d_transpose/StatefulPartitionedCall?
lambda_7/PartitionedCallPartitionedCall2conv_2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_7_layer_call_and_return_conditional_losses_46972932
lambda_7/PartitionedCall?
Un_Normalize/PartitionedCallPartitionedCall!lambda_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_46973142
Un_Normalize/PartitionedCall?
IdentityIdentity%Un_Normalize/PartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall*^conv_2d_transpose/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2V
)conv_2d_transpose/StatefulPartitionedCall)conv_2d_transpose/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_particle_autoencoder_layer_call_fn_4698443
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_46977812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1
?
?
4__inference_conv2d_transpose_1_layer_call_fn_4696912

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_46969022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?*
#__inference__traced_restore_4700626
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate$
 assignvariableop_5_conv2d_kernel"
assignvariableop_6_conv2d_bias$
 assignvariableop_7_conv1d_kernel"
assignvariableop_8_conv1d_bias&
"assignvariableop_9_conv1d_1_kernel%
!assignvariableop_10_conv1d_1_bias$
 assignvariableop_11_dense_kernel"
assignvariableop_12_dense_bias&
"assignvariableop_13_dense_1_kernel$
 assignvariableop_14_dense_1_bias 
assignvariableop_15_z_kernel
assignvariableop_16_z_bias&
"assignvariableop_17_dense_2_kernel$
 assignvariableop_18_dense_2_bias&
"assignvariableop_19_dense_3_kernel$
 assignvariableop_20_dense_3_bias&
"assignvariableop_21_dense_4_kernel$
 assignvariableop_22_dense_4_bias@
<assignvariableop_23_conv1d_transpose_conv2d_transpose_kernel>
:assignvariableop_24_conv1d_transpose_conv2d_transpose_biasD
@assignvariableop_25_conv1d_transpose_1_conv2d_transpose_1_kernelB
>assignvariableop_26_conv1d_transpose_1_conv2d_transpose_1_bias0
,assignvariableop_27_conv_2d_transpose_kernel.
*assignvariableop_28_conv_2d_transpose_bias,
(assignvariableop_29_adam_conv2d_kernel_m*
&assignvariableop_30_adam_conv2d_bias_m,
(assignvariableop_31_adam_conv1d_kernel_m*
&assignvariableop_32_adam_conv1d_bias_m.
*assignvariableop_33_adam_conv1d_1_kernel_m,
(assignvariableop_34_adam_conv1d_1_bias_m+
'assignvariableop_35_adam_dense_kernel_m)
%assignvariableop_36_adam_dense_bias_m-
)assignvariableop_37_adam_dense_1_kernel_m+
'assignvariableop_38_adam_dense_1_bias_m'
#assignvariableop_39_adam_z_kernel_m%
!assignvariableop_40_adam_z_bias_m-
)assignvariableop_41_adam_dense_2_kernel_m+
'assignvariableop_42_adam_dense_2_bias_m-
)assignvariableop_43_adam_dense_3_kernel_m+
'assignvariableop_44_adam_dense_3_bias_m-
)assignvariableop_45_adam_dense_4_kernel_m+
'assignvariableop_46_adam_dense_4_bias_mG
Cassignvariableop_47_adam_conv1d_transpose_conv2d_transpose_kernel_mE
Aassignvariableop_48_adam_conv1d_transpose_conv2d_transpose_bias_mK
Gassignvariableop_49_adam_conv1d_transpose_1_conv2d_transpose_1_kernel_mI
Eassignvariableop_50_adam_conv1d_transpose_1_conv2d_transpose_1_bias_m7
3assignvariableop_51_adam_conv_2d_transpose_kernel_m5
1assignvariableop_52_adam_conv_2d_transpose_bias_m,
(assignvariableop_53_adam_conv2d_kernel_v*
&assignvariableop_54_adam_conv2d_bias_v,
(assignvariableop_55_adam_conv1d_kernel_v*
&assignvariableop_56_adam_conv1d_bias_v.
*assignvariableop_57_adam_conv1d_1_kernel_v,
(assignvariableop_58_adam_conv1d_1_bias_v+
'assignvariableop_59_adam_dense_kernel_v)
%assignvariableop_60_adam_dense_bias_v-
)assignvariableop_61_adam_dense_1_kernel_v+
'assignvariableop_62_adam_dense_1_bias_v'
#assignvariableop_63_adam_z_kernel_v%
!assignvariableop_64_adam_z_bias_v-
)assignvariableop_65_adam_dense_2_kernel_v+
'assignvariableop_66_adam_dense_2_bias_v-
)assignvariableop_67_adam_dense_3_kernel_v+
'assignvariableop_68_adam_dense_3_bias_v-
)assignvariableop_69_adam_dense_4_kernel_v+
'assignvariableop_70_adam_dense_4_bias_vG
Cassignvariableop_71_adam_conv1d_transpose_conv2d_transpose_kernel_vE
Aassignvariableop_72_adam_conv1d_transpose_conv2d_transpose_bias_vK
Gassignvariableop_73_adam_conv1d_transpose_1_conv2d_transpose_1_kernel_vI
Eassignvariableop_74_adam_conv1d_transpose_1_conv2d_transpose_1_bias_v7
3assignvariableop_75_adam_conv_2d_transpose_kernel_v5
1assignvariableop_76_adam_conv_2d_transpose_bias_v
identity_78??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_8?AssignVariableOp_9?*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*?)
value?)B?)NB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*?
value?B?NB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*\
dtypesR
P2N	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_conv1d_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv1d_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv1d_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_dense_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_z_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_z_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_4_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp assignvariableop_22_dense_4_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp<assignvariableop_23_conv1d_transpose_conv2d_transpose_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp:assignvariableop_24_conv1d_transpose_conv2d_transpose_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp@assignvariableop_25_conv1d_transpose_1_conv2d_transpose_1_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp>assignvariableop_26_conv1d_transpose_1_conv2d_transpose_1_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_conv_2d_transpose_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_conv_2d_transpose_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv2d_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv1d_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv1d_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dense_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp#assignvariableop_39_adam_z_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp!assignvariableop_40_adam_z_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_3_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_3_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_4_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_4_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpCassignvariableop_47_adam_conv1d_transpose_conv2d_transpose_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpAassignvariableop_48_adam_conv1d_transpose_conv2d_transpose_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpGassignvariableop_49_adam_conv1d_transpose_1_conv2d_transpose_1_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpEassignvariableop_50_adam_conv1d_transpose_1_conv2d_transpose_1_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp3assignvariableop_51_adam_conv_2d_transpose_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp1assignvariableop_52_adam_conv_2d_transpose_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv2d_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_conv2d_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv1d_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_conv1d_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv1d_1_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv1d_1_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_dense_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp%assignvariableop_60_adam_dense_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp#assignvariableop_63_adam_z_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp!assignvariableop_64_adam_z_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp)assignvariableop_65_adam_dense_2_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp'assignvariableop_66_adam_dense_2_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_dense_3_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_dense_3_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_4_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_4_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpCassignvariableop_71_adam_conv1d_transpose_conv2d_transpose_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpAassignvariableop_72_adam_conv1d_transpose_conv2d_transpose_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOpGassignvariableop_73_adam_conv1d_transpose_1_conv2d_transpose_1_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOpEassignvariableop_74_adam_conv1d_transpose_1_conv2d_transpose_1_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp3assignvariableop_75_adam_conv_2d_transpose_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp1assignvariableop_76_adam_conv_2d_transpose_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_769
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_77?
Identity_78IdentityIdentity_77:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_78"#
identity_78Identity_78:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
C__inference_conv1d_layer_call_and_return_conditional_losses_4699722

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????b2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????`*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2	
BiasAddY
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
Elui
IdentityIdentityElu:activations:0*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????b:::S O
+
_output_shapes
:?????????b
 
_user_specified_nameinputs
?.
?
D__inference_encoder_layer_call_and_return_conditional_losses_4696700

inputs
conv2d_4696666
conv2d_4696668
conv1d_4696672
conv1d_4696674
conv1d_1_4696677
conv1d_1_4696679
dense_4696684
dense_4696686
dense_1_4696689
dense_1_4696691
	z_4696694
	z_4696696
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?z/StatefulPartitionedCall?
Std_Normalize/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_46963762
Std_Normalize/PartitionedCall?
lambda/PartitionedCallPartitionedCall&Std_Normalize/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_46963902
lambda/PartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_4696666conv2d_4696668*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_46964202 
conv2d/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_46964412
lambda_1/PartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_4696672conv1d_4696674*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_46964752 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_4696677conv1d_1_4696679*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_46965072"
 conv1d_1/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_layer_call_and_return_conditional_losses_46963582#
!average_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall*average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_46965302
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4696684dense_4696686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_46965492
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4696689dense_1_4696691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_46965762!
dense_1/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0	z_4696694	z_4696696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_z_layer_call_and_return_conditional_losses_46966022
z/StatefulPartitionedCall?
IdentityIdentity"z/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
)__inference_encoder_layer_call_fn_4699230

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_46967002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????d::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?%
?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4696902

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Elu
IdentityIdentityElu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
)__inference_decoder_layer_call_fn_4699602

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46974042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
N__inference_conv_2d_transpose_layer_call_and_return_conditional_losses_4696950

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????d@
output_14
StatefulPartitionedCall:0?????????dtensorflow/serving/predict:??
?
shape_convolved
encoder
decoder
	optimizer
loss
trainable_variables
regularization_losses
	variables
		keras_api


signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
?	reco_loss"?
_tf_keras_model?{"class_name": "ParticleAutoencoder", "name": "particle_autoencoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ParticleAutoencoder"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 2.220446154715841e-19, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
 "
trackable_list_wrapper
?g
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer_with_weights-3
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?c
_tf_keras_network?c{"class_name": "Functional", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "StdNormalization", "config": {"name": "Std_Normalize", "trainable": false, "dtype": "float32", "mean_x": [0.00048135186079889536, -2.844892514985986e-05, 3.404677391052246], "std_x": [0.1998933106660843, 0.2044232338666916, 18.272403717041016]}, "name": "Std_Normalize", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukDAAAAKQHaBGF4aXMp\nAtoCdGbaC2V4cGFuZF9kaW1zKQHaAXipAHIGAAAA+lYvZW9zL2hvbWUtay9raXdvem5pYS9kZXYv\nbGF0ZW50X3NwYWNlX2NsdXN0ZXJpbmdfZm9yX2FkL2xhc3BhY2x1L21vZGVscy9hdXRvZW5jb2Rl\nci5wedoIPGxhbWJkYT4hAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["Std_Normalize", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaB3NxdWVlemUpAdoBeKkAcgYAAAD6Vi9lb3MvaG9tZS1rL2tpd296bmlhL2Rldi9sYXRl\nbnRfc3BhY2VfY2x1c3RlcmluZ19mb3JfYWQvbGFzcGFjbHUvbW9kZWxzL2F1dG9lbmNvZGVyLnB5\n2gg8bGFtYmRhPiUAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["lambda_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 102, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 24, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["z", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "StdNormalization", "config": {"name": "Std_Normalize", "trainable": false, "dtype": "float32", "mean_x": [0.00048135186079889536, -2.844892514985986e-05, 3.404677391052246], "std_x": [0.1998933106660843, 0.2044232338666916, 18.272403717041016]}, "name": "Std_Normalize", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukDAAAAKQHaBGF4aXMp\nAtoCdGbaC2V4cGFuZF9kaW1zKQHaAXipAHIGAAAA+lYvZW9zL2hvbWUtay9raXdvem5pYS9kZXYv\nbGF0ZW50X3NwYWNlX2NsdXN0ZXJpbmdfZm9yX2FkL2xhc3BhY2x1L21vZGVscy9hdXRvZW5jb2Rl\nci5wedoIPGxhbWJkYT4hAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["Std_Normalize", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaB3NxdWVlemUpAdoBeKkAcgYAAAD6Vi9lb3MvaG9tZS1rL2tpd296bmlhL2Rldi9sYXRl\nbnRfc3BhY2VfY2x1c3RlcmluZ19mb3JfYWQvbGFzcGFjbHUvbW9kZWxzL2F1dG9lbmNvZGVyLnB5\n2gg8bGFtYmRhPiUAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["lambda_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 102, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 24, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["z", 0, 0]]}}}
?W
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
 layer-5
!layer_with_weights-3
!layer-6
"layer_with_weights-4
"layer-7
#layer-8
$layer_with_weights-5
$layer-9
%layer-10
&layer-11
'trainable_variables
(regularization_losses
)	variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?T
_tf_keras_network?T{"class_name": "Functional", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z"}, "name": "z", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 24, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["z", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 102, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [47, 24]}}, "name": "reshape", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "UpSampling1D", "config": {"name": "up_sampling1d", "trainable": true, "dtype": "float32", "size": 2}, "name": "up_sampling1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "kernel_sz": 3, "filters": 20, "activation": "elu", "kernel_initializer": "he_uniform"}, "name": "conv1d_transpose", "inbound_nodes": [[["up_sampling1d", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_1", "trainable": true, "dtype": "float32", "kernel_sz": 3, "filters": 16, "activation": "elu", "kernel_initializer": "he_uniform"}, "name": "conv1d_transpose_1", "inbound_nodes": [[["conv1d_transpose", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaC2V4cGFuZF9kaW1zKQHaAXipAHIGAAAA+lYvZW9zL2hvbWUtay9raXdvem5pYS9kZXYv\nbGF0ZW50X3NwYWNlX2NsdXN0ZXJpbmdfZm9yX2FkL2xhc3BhY2x1L21vZGVscy9hdXRvZW5jb2Rl\nci5wedoIPGxhbWJkYT5QAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_6", "inbound_nodes": [[["conv1d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_2d_transpose", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_2d_transpose", "inbound_nodes": [[["lambda_6", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukDAAAAKQHaBGF4aXMp\nAtoCdGbaB3NxdWVlemUpAdoBeKkAcgYAAAD6Vi9lb3MvaG9tZS1rL2tpd296bmlhL2Rldi9sYXRl\nbnRfc3BhY2VfY2x1c3RlcmluZ19mb3JfYWQvbGFzcGFjbHUvbW9kZWxzL2F1dG9lbmNvZGVyLnB5\n2gg8bGFtYmRhPlMAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_7", "inbound_nodes": [[["conv_2d_transpose", 0, 0, {}]]]}, {"class_name": "StdUnnormalization", "config": {"name": "Un_Normalize", "trainable": false, "dtype": "float32", "mean_x": [0.00048135186079889536, -2.844892514985986e-05, 3.404677391052246], "std_x": [0.1998933106660843, 0.2044232338666916, 18.272403717041016]}, "name": "Un_Normalize", "inbound_nodes": [[["lambda_7", 0, 0, {}]]]}], "input_layers": [["z", 0, 0]], "output_layers": [["Un_Normalize", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z"}, "name": "z", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 24, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["z", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 102, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [47, 24]}}, "name": "reshape", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "UpSampling1D", "config": {"name": "up_sampling1d", "trainable": true, "dtype": "float32", "size": 2}, "name": "up_sampling1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "kernel_sz": 3, "filters": 20, "activation": "elu", "kernel_initializer": "he_uniform"}, "name": "conv1d_transpose", "inbound_nodes": [[["up_sampling1d", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_1", "trainable": true, "dtype": "float32", "kernel_sz": 3, "filters": 16, "activation": "elu", "kernel_initializer": "he_uniform"}, "name": "conv1d_transpose_1", "inbound_nodes": [[["conv1d_transpose", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaC2V4cGFuZF9kaW1zKQHaAXipAHIGAAAA+lYvZW9zL2hvbWUtay9raXdvem5pYS9kZXYv\nbGF0ZW50X3NwYWNlX2NsdXN0ZXJpbmdfZm9yX2FkL2xhc3BhY2x1L21vZGVscy9hdXRvZW5jb2Rl\nci5wedoIPGxhbWJkYT5QAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_6", "inbound_nodes": [[["conv1d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_2d_transpose", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_2d_transpose", "inbound_nodes": [[["lambda_6", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukDAAAAKQHaBGF4aXMp\nAtoCdGbaB3NxdWVlemUpAdoBeKkAcgYAAAD6Vi9lb3MvaG9tZS1rL2tpd296bmlhL2Rldi9sYXRl\nbnRfc3BhY2VfY2x1c3RlcmluZ19mb3JfYWQvbGFzcGFjbHUvbW9kZWxzL2F1dG9lbmNvZGVyLnB5\n2gg8bGFtYmRhPlMAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_7", "inbound_nodes": [[["conv_2d_transpose", 0, 0, {}]]]}, {"class_name": "StdUnnormalization", "config": {"name": "Un_Normalize", "trainable": false, "dtype": "float32", "mean_x": [0.00048135186079889536, -2.844892514985986e-05, 3.404677391052246], "std_x": [0.1998933106660843, 0.2044232338666916, 18.272403717041016]}, "name": "Un_Normalize", "inbound_nodes": [[["lambda_7", 0, 0, {}]]]}], "input_layers": [["z", 0, 0]], "output_layers": [["Un_Normalize", 0, 0]]}}}
?
+iter

,beta_1

-beta_2
	.decay
/learning_rate0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?"
	optimizer
 "
trackable_dict_wrapper
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23"
trackable_list_wrapper
?
trainable_variables

Hlayers
Ilayer_metrics
Jnon_trainable_variables
Kmetrics
Llayer_regularization_losses
regularization_losses
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
?
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "StdNormalization", "name": "Std_Normalize", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Std_Normalize", "trainable": false, "dtype": "float32", "mean_x": [0.00048135186079889536, -2.844892514985986e-05, 3.404677391052246], "std_x": [0.1998933106660843, 0.2044232338666916, 18.272403717041016]}}
?
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukDAAAAKQHaBGF4aXMp\nAtoCdGbaC2V4cGFuZF9kaW1zKQHaAXipAHIGAAAA+lYvZW9zL2hvbWUtay9raXdvem5pYS9kZXYv\nbGF0ZW50X3NwYWNlX2NsdXN0ZXJpbmdfZm9yX2FkL2xhc3BhY2x1L21vZGVscy9hdXRvZW5jb2Rl\nci5wedoIPGxhbWJkYT4hAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?	

0kernel
1bias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 3, 1]}}
?
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaB3NxdWVlemUpAdoBeKkAcgYAAAD6Vi9lb3MvaG9tZS1rL2tpd296bmlhL2Rldi9sYXRl\nbnRfc3BhY2VfY2x1c3RlcmluZ19mb3JfYWQvbGFzcGFjbHUvbW9kZWxzL2F1dG9lbmNvZGVyLnB5\n2gg8bGFtYmRhPiUAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?	

2kernel
3bias
]trainable_variables
^regularization_losses
_	variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 98, 16]}}
?	

4kernel
5bias
atrainable_variables
bregularization_losses
c	variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 20]}}
?
etrainable_variables
fregularization_losses
g	variables
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AveragePooling1D", "name": "average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
itrainable_variables
jregularization_losses
k	variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

6kernel
7bias
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 102, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1128]}}
?

8kernel
9bias
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 24, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 102}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 102]}}
?

:kernel
;bias
utrainable_variables
vregularization_losses
w	variables
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "z", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
v
00
11
22
33
44
55
66
77
88
99
:10
;11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
00
11
22
33
44
55
66
77
88
99
:10
;11"
trackable_list_wrapper
?
trainable_variables

ylayers
zlayer_metrics
{non_trainable_variables
|metrics
}layer_regularization_losses
regularization_losses
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "z", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z"}}
?

<kernel
=bias
~trainable_variables
regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 24, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
?

>kernel
?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 102, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
?

@kernel
Abias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 102}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 102]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [47, 24]}}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling1D", "name": "up_sampling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling1d", "trainable": true, "dtype": "float32", "size": 2}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?ExpandChannel
?ConvTranspose
?SqueezeChannel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "kernel_sz": 3, "filters": 20, "activation": "elu", "kernel_initializer": "he_uniform"}}
?
?ExpandChannel
?ConvTranspose
?SqueezeChannel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_1", "trainable": true, "dtype": "float32", "kernel_sz": 3, "filters": 16, "activation": "elu", "kernel_initializer": "he_uniform"}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaC2V4cGFuZF9kaW1zKQHaAXipAHIGAAAA+lYvZW9zL2hvbWUtay9raXdvem5pYS9kZXYv\nbGF0ZW50X3NwYWNlX2NsdXN0ZXJpbmdfZm9yX2FkL2xhc3BhY2x1L21vZGVscy9hdXRvZW5jb2Rl\nci5wedoIPGxhbWJkYT5QAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?


Fkernel
Gbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv_2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_2d_transpose", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 98, 1, 16]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukDAAAAKQHaBGF4aXMp\nAtoCdGbaB3NxdWVlemUpAdoBeKkAcgYAAAD6Vi9lb3MvaG9tZS1rL2tpd296bmlhL2Rldi9sYXRl\nbnRfc3BhY2VfY2x1c3RlcmluZ19mb3JfYWQvbGFzcGFjbHUvbW9kZWxzL2F1dG9lbmNvZGVyLnB5\n2gg8bGFtYmRhPlMAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models.autoencoder", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "StdUnnormalization", "name": "Un_Normalize", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Un_Normalize", "trainable": false, "dtype": "float32", "mean_x": [0.00048135186079889536, -2.844892514985986e-05, 3.404677391052246], "std_x": [0.1998933106660843, 0.2044232338666916, 18.272403717041016]}}
v
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11"
trackable_list_wrapper
?
'trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
(regularization_losses
)	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%2conv2d/kernel
:2conv2d/bias
#:!2conv1d/kernel
:2conv1d/bias
%:#2conv1d_1/kernel
:2conv1d_1/bias
:	?f2dense/kernel
:f2
dense/bias
 :f2dense_1/kernel
:2dense_1/bias
:2z/kernel
:2z/bias
 :2dense_2/kernel
:2dense_2/bias
 :f2dense_3/kernel
:f2dense_3/bias
!:	f?2dense_4/kernel
:?2dense_4/bias
B:@2(conv1d_transpose/conv2d_transpose/kernel
4:22&conv1d_transpose/conv2d_transpose/bias
F:D2,conv1d_transpose_1/conv2d_transpose_1/kernel
8:62*conv1d_transpose_1/conv2d_transpose_1/bias
2:02conv_2d_transpose/kernel
$:"2conv_2d_transpose/bias
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mtrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
Nregularization_losses
O	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qtrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
Rregularization_losses
S	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
Utrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
Vregularization_losses
W	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ytrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
Zregularization_losses
[	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
]trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
^regularization_losses
_	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
atrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
bregularization_losses
c	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
etrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
fregularization_losses
g	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
itrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
jregularization_losses
k	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
mtrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
nregularization_losses
o	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
qtrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
rregularization_losses
s	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
utrainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
vregularization_losses
w	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
~trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaC2V4cGFuZF9kaW1zKQHaAXipAHIGAAAA+kQvZW9zL2hvbWUtay9raXdvem5pYS9kZXYv\nYXV0b2VuY29kZXJfZm9yX2Fub21hbHkvdmFuZGUvdmFlL2xheWVycy5wedoIPGxhbWJkYT4hAAAA\n8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "vande.vae.layers", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?


Bkernel
Cbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94, 1, 24]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaB3NxdWVlemUpAdoBeKkAcgYAAAD6RC9lb3MvaG9tZS1rL2tpd296bmlhL2Rldi9hdXRv\nZW5jb2Rlcl9mb3JfYW5vbWFseS92YW5kZS92YWUvbGF5ZXJzLnB52gg8bGFtYmRhPiMAAADzAAAA\nAA==\n", null, null]}, "function_type": "lambda", "module": "vande.vae.layers", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaC2V4cGFuZF9kaW1zKQHaAXipAHIGAAAA+kQvZW9zL2hvbWUtay9raXdvem5pYS9kZXYv\nYXV0b2VuY29kZXJfZm9yX2Fub21hbHkvdmFuZGUvdmFlL2xheWVycy5wedoIPGxhbWJkYT4hAAAA\n8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "vande.vae.layers", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?


Dkernel
Ebias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 1, 20]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkAWQCjQJTACkDTukCAAAAKQHaBGF4aXMp\nAtoCdGbaB3NxdWVlemUpAdoBeKkAcgYAAAD6RC9lb3MvaG9tZS1rL2tpd296bmlhL2Rldi9hdXRv\nZW5jb2Rlcl9mb3JfYW5vbWFseS92YW5kZS92YWUvbGF5ZXJzLnB52gg8bGFtYmRhPiMAAADzAAAA\nAA==\n", null, null]}, "function_type": "lambda", "module": "vande.vae.layers", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
v
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
(:&2Adam/conv1d/kernel/m
:2Adam/conv1d/bias/m
*:(2Adam/conv1d_1/kernel/m
 :2Adam/conv1d_1/bias/m
$:"	?f2Adam/dense/kernel/m
:f2Adam/dense/bias/m
%:#f2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
:2Adam/z/kernel/m
:2Adam/z/bias/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#f2Adam/dense_3/kernel/m
:f2Adam/dense_3/bias/m
&:$	f?2Adam/dense_4/kernel/m
 :?2Adam/dense_4/bias/m
G:E2/Adam/conv1d_transpose/conv2d_transpose/kernel/m
9:72-Adam/conv1d_transpose/conv2d_transpose/bias/m
K:I23Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/m
=:;21Adam/conv1d_transpose_1/conv2d_transpose_1/bias/m
7:52Adam/conv_2d_transpose/kernel/m
):'2Adam/conv_2d_transpose/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
(:&2Adam/conv1d/kernel/v
:2Adam/conv1d/bias/v
*:(2Adam/conv1d_1/kernel/v
 :2Adam/conv1d_1/bias/v
$:"	?f2Adam/dense/kernel/v
:f2Adam/dense/bias/v
%:#f2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
:2Adam/z/kernel/v
:2Adam/z/bias/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:#f2Adam/dense_3/kernel/v
:f2Adam/dense_3/bias/v
&:$	f?2Adam/dense_4/kernel/v
 :?2Adam/dense_4/bias/v
G:E2/Adam/conv1d_transpose/conv2d_transpose/kernel/v
9:72-Adam/conv1d_transpose/conv2d_transpose/bias/v
K:I23Adam/conv1d_transpose_1/conv2d_transpose_1/kernel/v
=:;21Adam/conv1d_transpose_1/conv2d_transpose_1/bias/v
7:52Adam/conv_2d_transpose/kernel/v
):'2Adam/conv_2d_transpose/bias/v
?2?
6__inference_particle_autoencoder_layer_call_fn_4699044
6__inference_particle_autoencoder_layer_call_fn_4698496
6__inference_particle_autoencoder_layer_call_fn_4698443
6__inference_particle_autoencoder_layer_call_fn_4698991?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference__wrapped_model_4696349?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
input_1?????????d
?2?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698938
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698390
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698169
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698717?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_threeD_loss_4699065?
???
FullArgSpec 
args?
jinputs
	joutputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_encoder_layer_call_fn_4696727
)__inference_encoder_layer_call_fn_4699259
)__inference_encoder_layer_call_fn_4699230
)__inference_encoder_layer_call_fn_4696795?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_encoder_layer_call_and_return_conditional_losses_4696619
D__inference_encoder_layer_call_and_return_conditional_losses_4699201
D__inference_encoder_layer_call_and_return_conditional_losses_4699133
D__inference_encoder_layer_call_and_return_conditional_losses_4696658?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_decoder_layer_call_fn_4699602
)__inference_decoder_layer_call_fn_4697499
)__inference_decoder_layer_call_fn_4697431
)__inference_decoder_layer_call_fn_4699631?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_decoder_layer_call_and_return_conditional_losses_4699416
D__inference_decoder_layer_call_and_return_conditional_losses_4699573
D__inference_decoder_layer_call_and_return_conditional_losses_4697323
D__inference_decoder_layer_call_and_return_conditional_losses_4697362?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
4B2
%__inference_signature_wrapper_4697948input_1
?2?
/__inference_Std_Normalize_layer_call_fn_4699644?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_4699639?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_lambda_layer_call_fn_4699661
(__inference_lambda_layer_call_fn_4699666?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_lambda_layer_call_and_return_conditional_losses_4699656
C__inference_lambda_layer_call_and_return_conditional_losses_4699650?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv2d_layer_call_fn_4699686?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_layer_call_and_return_conditional_losses_4699677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_lambda_1_layer_call_fn_4699706
*__inference_lambda_1_layer_call_fn_4699701?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_lambda_1_layer_call_and_return_conditional_losses_4699691
E__inference_lambda_1_layer_call_and_return_conditional_losses_4699696?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv1d_layer_call_fn_4699731?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1d_layer_call_and_return_conditional_losses_4699722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_1_layer_call_fn_4699756?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_1_layer_call_and_return_conditional_losses_4699747?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_average_pooling1d_layer_call_fn_4696364?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
N__inference_average_pooling1d_layer_call_and_return_conditional_losses_4696358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
)__inference_flatten_layer_call_fn_4699767?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_4699762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_4699787?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_4699778?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_4699807?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_4699798?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_z_layer_call_fn_4699826?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_z_layer_call_and_return_conditional_losses_4699817?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_2_layer_call_fn_4699846?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_2_layer_call_and_return_conditional_losses_4699837?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_3_layer_call_fn_4699866?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_3_layer_call_and_return_conditional_losses_4699857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_4_layer_call_fn_4699886?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_4_layer_call_and_return_conditional_losses_4699877?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_layer_call_fn_4699904?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_layer_call_and_return_conditional_losses_4699899?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_up_sampling1d_layer_call_fn_4696814?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
J__inference_up_sampling1d_layer_call_and_return_conditional_losses_4696808?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
2__inference_conv1d_transpose_layer_call_fn_4699990
2__inference_conv1d_transpose_layer_call_fn_4699981?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_4699938
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_4699972?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv1d_transpose_1_layer_call_fn_4700076
4__inference_conv1d_transpose_1_layer_call_fn_4700067?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_4700058
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_4700024?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_lambda_6_layer_call_fn_4700093
*__inference_lambda_6_layer_call_fn_4700098?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_lambda_6_layer_call_and_return_conditional_losses_4700082
E__inference_lambda_6_layer_call_and_return_conditional_losses_4700088?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_conv_2d_transpose_layer_call_fn_4696960?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
N__inference_conv_2d_transpose_layer_call_and_return_conditional_losses_4696950?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
*__inference_lambda_7_layer_call_fn_4700113
*__inference_lambda_7_layer_call_fn_4700118?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_lambda_7_layer_call_and_return_conditional_losses_4700103
E__inference_lambda_7_layer_call_and_return_conditional_losses_4700108?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_Un_Normalize_layer_call_fn_4700131?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_4700126?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_conv2d_transpose_layer_call_fn_4696863?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4696853?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_conv2d_transpose_1_layer_call_fn_4696912?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4696902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
J__inference_Std_Normalize_layer_call_and_return_conditional_losses_4699639[.?+
$?!
?
x?????????d
? ")?&
?
0?????????d
? ?
/__inference_Std_Normalize_layer_call_fn_4699644N.?+
$?!
?
x?????????d
? "??????????d?
I__inference_Un_Normalize_layer_call_and_return_conditional_losses_4700126v@?=
6?3
1?.
x'???????????????????????????
? "2?/
(?%
0??????????????????
? ?
.__inference_Un_Normalize_layer_call_fn_4700131i@?=
6?3
1?.
x'???????????????????????????
? "%?"???????????????????
"__inference__wrapped_model_4696349?0123456789:;<=>?@ABCDEFG4?1
*?'
%?"
input_1?????????d
? "7?4
2
output_1&?#
output_1?????????d?
N__inference_average_pooling1d_layer_call_and_return_conditional_losses_4696358?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
3__inference_average_pooling1d_layer_call_fn_4696364wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
E__inference_conv1d_1_layer_call_and_return_conditional_losses_4699747d453?0
)?&
$?!
inputs?????????`
? ")?&
?
0?????????^
? ?
*__inference_conv1d_1_layer_call_fn_4699756W453?0
)?&
$?!
inputs?????????`
? "??????????^?
C__inference_conv1d_layer_call_and_return_conditional_losses_4699722d233?0
)?&
$?!
inputs?????????b
? ")?&
?
0?????????`
? ?
(__inference_conv1d_layer_call_fn_4699731W233?0
)?&
$?!
inputs?????????b
? "??????????`?
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_4700024zDE@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
O__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_4700058zDE@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
4__inference_conv1d_transpose_1_layer_call_fn_4700067mDE@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
4__inference_conv1d_transpose_1_layer_call_fn_4700076mDE@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_4699938?BCI?F
??<
6?3
inputs'???????????????????????????
p
? "2?/
(?%
0??????????????????
? ?
M__inference_conv1d_transpose_layer_call_and_return_conditional_losses_4699972?BCI?F
??<
6?3
inputs'???????????????????????????
p 
? "2?/
(?%
0??????????????????
? ?
2__inference_conv1d_transpose_layer_call_fn_4699981vBCI?F
??<
6?3
inputs'???????????????????????????
p
? "%?"???????????????????
2__inference_conv1d_transpose_layer_call_fn_4699990vBCI?F
??<
6?3
inputs'???????????????????????????
p 
? "%?"???????????????????
C__inference_conv2d_layer_call_and_return_conditional_losses_4699677l017?4
-?*
(?%
inputs?????????d
? "-?*
#? 
0?????????b
? ?
(__inference_conv2d_layer_call_fn_4699686_017?4
-?*
(?%
inputs?????????d
? " ??????????b?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4696902?DEI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
4__inference_conv2d_transpose_1_layer_call_fn_4696912?DEI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4696853?BCI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
2__inference_conv2d_transpose_layer_call_fn_4696863?BCI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
N__inference_conv_2d_transpose_layer_call_and_return_conditional_losses_4696950?FGI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv_2d_transpose_layer_call_fn_4696960?FGI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
D__inference_decoder_layer_call_and_return_conditional_losses_4697323v<=>?@ABCDEFG2?/
(?%
?
z?????????
p

 
? "2?/
(?%
0??????????????????
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_4697362v<=>?@ABCDEFG2?/
(?%
?
z?????????
p 

 
? "2?/
(?%
0??????????????????
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_4699416r<=>?@ABCDEFG7?4
-?*
 ?
inputs?????????
p

 
? ")?&
?
0?????????d
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_4699573r<=>?@ABCDEFG7?4
-?*
 ?
inputs?????????
p 

 
? ")?&
?
0?????????d
? ?
)__inference_decoder_layer_call_fn_4697431i<=>?@ABCDEFG2?/
(?%
?
z?????????
p

 
? "%?"???????????????????
)__inference_decoder_layer_call_fn_4697499i<=>?@ABCDEFG2?/
(?%
?
z?????????
p 

 
? "%?"???????????????????
)__inference_decoder_layer_call_fn_4699602n<=>?@ABCDEFG7?4
-?*
 ?
inputs?????????
p

 
? "%?"???????????????????
)__inference_decoder_layer_call_fn_4699631n<=>?@ABCDEFG7?4
-?*
 ?
inputs?????????
p 

 
? "%?"???????????????????
D__inference_dense_1_layer_call_and_return_conditional_losses_4699798\89/?,
%?"
 ?
inputs?????????f
? "%?"
?
0?????????
? |
)__inference_dense_1_layer_call_fn_4699807O89/?,
%?"
 ?
inputs?????????f
? "???????????
D__inference_dense_2_layer_call_and_return_conditional_losses_4699837\<=/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_2_layer_call_fn_4699846O<=/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dense_3_layer_call_and_return_conditional_losses_4699857\>?/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????f
? |
)__inference_dense_3_layer_call_fn_4699866O>?/?,
%?"
 ?
inputs?????????
? "??????????f?
D__inference_dense_4_layer_call_and_return_conditional_losses_4699877]@A/?,
%?"
 ?
inputs?????????f
? "&?#
?
0??????????
? }
)__inference_dense_4_layer_call_fn_4699886P@A/?,
%?"
 ?
inputs?????????f
? "????????????
B__inference_dense_layer_call_and_return_conditional_losses_4699778]670?-
&?#
!?
inputs??????????
? "%?"
?
0?????????f
? {
'__inference_dense_layer_call_fn_4699787P670?-
&?#
!?
inputs??????????
? "??????????f?
D__inference_encoder_layer_call_and_return_conditional_losses_4696619y0123456789:;B??
8?5
+?(
encoder_input?????????d
p

 
? "%?"
?
0?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_4696658y0123456789:;B??
8?5
+?(
encoder_input?????????d
p 

 
? "%?"
?
0?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_4699133r0123456789:;;?8
1?.
$?!
inputs?????????d
p

 
? "%?"
?
0?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_4699201r0123456789:;;?8
1?.
$?!
inputs?????????d
p 

 
? "%?"
?
0?????????
? ?
)__inference_encoder_layer_call_fn_4696727l0123456789:;B??
8?5
+?(
encoder_input?????????d
p

 
? "???????????
)__inference_encoder_layer_call_fn_4696795l0123456789:;B??
8?5
+?(
encoder_input?????????d
p 

 
? "???????????
)__inference_encoder_layer_call_fn_4699230e0123456789:;;?8
1?.
$?!
inputs?????????d
p

 
? "???????????
)__inference_encoder_layer_call_fn_4699259e0123456789:;;?8
1?.
$?!
inputs?????????d
p 

 
? "???????????
D__inference_flatten_layer_call_and_return_conditional_losses_4699762]3?0
)?&
$?!
inputs?????????/
? "&?#
?
0??????????
? }
)__inference_flatten_layer_call_fn_4699767P3?0
)?&
$?!
inputs?????????/
? "????????????
E__inference_lambda_1_layer_call_and_return_conditional_losses_4699691l??<
5?2
(?%
inputs?????????b

 
p
? ")?&
?
0?????????b
? ?
E__inference_lambda_1_layer_call_and_return_conditional_losses_4699696l??<
5?2
(?%
inputs?????????b

 
p 
? ")?&
?
0?????????b
? ?
*__inference_lambda_1_layer_call_fn_4699701_??<
5?2
(?%
inputs?????????b

 
p
? "??????????b?
*__inference_lambda_1_layer_call_fn_4699706_??<
5?2
(?%
inputs?????????b

 
p 
? "??????????b?
E__inference_lambda_6_layer_call_and_return_conditional_losses_4700082~D?A
:?7
-?*
inputs??????????????????

 
p
? "6?3
,?)
0"??????????????????
? ?
E__inference_lambda_6_layer_call_and_return_conditional_losses_4700088~D?A
:?7
-?*
inputs??????????????????

 
p 
? "6?3
,?)
0"??????????????????
? ?
*__inference_lambda_6_layer_call_fn_4700093qD?A
:?7
-?*
inputs??????????????????

 
p
? ")?&"???????????????????
*__inference_lambda_6_layer_call_fn_4700098qD?A
:?7
-?*
inputs??????????????????

 
p 
? ")?&"???????????????????
E__inference_lambda_7_layer_call_and_return_conditional_losses_4700103?Q?N
G?D
:?7
inputs+???????????????????????????

 
p
? ";?8
1?.
0'???????????????????????????
? ?
E__inference_lambda_7_layer_call_and_return_conditional_losses_4700108?Q?N
G?D
:?7
inputs+???????????????????????????

 
p 
? ";?8
1?.
0'???????????????????????????
? ?
*__inference_lambda_7_layer_call_fn_4700113?Q?N
G?D
:?7
inputs+???????????????????????????

 
p
? ".?+'????????????????????????????
*__inference_lambda_7_layer_call_fn_4700118?Q?N
G?D
:?7
inputs+???????????????????????????

 
p 
? ".?+'????????????????????????????
C__inference_lambda_layer_call_and_return_conditional_losses_4699650l;?8
1?.
$?!
inputs?????????d

 
p
? "-?*
#? 
0?????????d
? ?
C__inference_lambda_layer_call_and_return_conditional_losses_4699656l;?8
1?.
$?!
inputs?????????d

 
p 
? "-?*
#? 
0?????????d
? ?
(__inference_lambda_layer_call_fn_4699661_;?8
1?.
$?!
inputs?????????d

 
p
? " ??????????d?
(__inference_lambda_layer_call_fn_4699666_;?8
1?.
$?!
inputs?????????d

 
p 
? " ??????????d?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_46981690123456789:;<=>?@ABCDEFG8?5
.?+
%?"
input_1?????????d
p
? ")?&
?
0?????????d
? ?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_46983900123456789:;<=>?@ABCDEFG8?5
.?+
%?"
input_1?????????d
p 
? ")?&
?
0?????????d
? ?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698717y0123456789:;<=>?@ABCDEFG2?/
(?%
?
x?????????d
p
? ")?&
?
0?????????d
? ?
Q__inference_particle_autoencoder_layer_call_and_return_conditional_losses_4698938y0123456789:;<=>?@ABCDEFG2?/
(?%
?
x?????????d
p 
? ")?&
?
0?????????d
? ?
6__inference_particle_autoencoder_layer_call_fn_4698443{0123456789:;<=>?@ABCDEFG8?5
.?+
%?"
input_1?????????d
p
? "%?"???????????????????
6__inference_particle_autoencoder_layer_call_fn_4698496{0123456789:;<=>?@ABCDEFG8?5
.?+
%?"
input_1?????????d
p 
? "%?"???????????????????
6__inference_particle_autoencoder_layer_call_fn_4698991u0123456789:;<=>?@ABCDEFG2?/
(?%
?
x?????????d
p
? "%?"???????????????????
6__inference_particle_autoencoder_layer_call_fn_4699044u0123456789:;<=>?@ABCDEFG2?/
(?%
?
x?????????d
p 
? "%?"???????????????????
D__inference_reshape_layer_call_and_return_conditional_losses_4699899]0?-
&?#
!?
inputs??????????
? ")?&
?
0?????????/
? }
)__inference_reshape_layer_call_fn_4699904P0?-
&?#
!?
inputs??????????
? "??????????/?
%__inference_signature_wrapper_4697948?0123456789:;<=>?@ABCDEFG??<
? 
5?2
0
input_1%?"
input_1?????????d"7?4
2
output_1&?#
output_1?????????d}
__inference_threeD_loss_4699065ZJ?G
@?=
?
inputs?d
?
outputs?d
? "?	??
J__inference_up_sampling1d_layer_call_and_return_conditional_losses_4696808?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
/__inference_up_sampling1d_layer_call_fn_4696814wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
>__inference_z_layer_call_and_return_conditional_losses_4699817\:;/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_z_layer_call_fn_4699826O:;/?,
%?"
 ?
inputs?????????
? "??????????