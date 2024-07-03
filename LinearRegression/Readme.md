### 线性模型的表示

线性模型可以用矩阵的形式表示如下：
$$ \mathbf{y} = \mathbf{X} \boldsymbol{\theta} $$
这里：
- $\mathbf{y} $ 是一个 $ n \times 1 $ 的向量，代表目标值。
- $ \mathbf{X} $ 是一个 $ n \times (p+1) $ 的矩阵，其中包括每个样本的特征，并且通常第一列为全1，用于模型中的截距项。
- $ \boldsymbol{\theta} $ 是一个 $ (p+1) \times 1 $ 的向量，包含模型的参数，即斜率和截距。

### 二范数损失函数

二范数损失函数，也称为平方损失函数，用矩阵形式表示为：
$$ L(\boldsymbol{\theta}) = (\mathbf{y} - \mathbf{X} \boldsymbol{\theta})^\top (\mathbf{y} - \mathbf{X} \boldsymbol{\theta}) $$
这里的损失函数是预测误差的平方和，用于衡量模型预测与实际数据之间的差距。

### 损失函数的梯度

损失函数关于参数 $ \boldsymbol{\theta} $ 的梯度可以表示为：
$$ \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) = -2 \mathbf{X}^\top (\mathbf{y} - \mathbf{X} \boldsymbol{\theta}) $$
这个梯度用于在梯度下降算法中更新参数 $ \boldsymbol{\theta} $。

### 梯度下降更新规则

参数的更新规则在梯度下降中通过以下公式实现：
$$ \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) $$
其中 $ \alpha $ 是学习率，决定了参数更新的步长大小。
