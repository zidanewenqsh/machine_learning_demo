## 损失函数
逻辑回归通常使用的损失函数是**交叉熵损失**，它用于衡量模型预测概率分布与实际标签之间的差异。对于给定的数据集 \(\{(x^{(i)}, y^{(i)})\}\)，其中 \(y^{(i)}\) 是实际类别标签（0或1），模型的损失函数 \(J(\theta)\) 可以表示为：

\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]
\]

这里，\(\hat{p}^{(i)}\) 是模型预测样本 \(x^{(i)}\) 属于类别1的概率，计算公式为：

\[
\hat{p}^{(i)} = \sigma(\theta^T x^{(i)})
\]

其中，\(\sigma\) 是sigmoid函数，\(\theta\) 是模型参数，\(\theta^T x^{(i)}\) 是参数和特征向量的点积。

### Sigmoid函数
Sigmoid函数定义为：

\[
\sigma(t) = \frac{1}{1 + e^{-t}}
\]

### 梯度计算
为了使用梯度下降优化损失函数，我们需要计算损失函数 \(J(\theta)\) 对于每个参数 \(\theta_j\) 的偏导数。这个偏导数给出了损失函数在参数 \(\theta_j\) 方向上的梯度。对于参数 \(\theta_j\)，其梯度计算公式为：

\[
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^m \left(\sigma(\theta^T x^{(i)}) - y^{(i)}\right) x_j^{(i)}
\]

这里，\(x_j^{(i)}\) 是第 \(i\) 个样本的第 \(j\) 个特征。

### 梯度下降更新规则
在计算了梯度之后，参数 \(\theta\) 通过以下规则更新，以最小化损失函数：

\[
\theta = \theta - \alpha \cdot \nabla_\theta J(\theta)
\]

这里，\(\alpha\) 是学习率，\(\nabla_\theta J(\theta)\) 是损失函数的梯度向量，包含了所有参数的梯度。

通过重复这个更新步骤（迭代），梯度下降方法可以逐渐找到损失函数的最小值，从而找到最佳的参数集合 \(\theta\)，使得模型在给定的数据上表现最好。






### 决策边界的数学表述

在逻辑回归中，模型预测的是样本属于某一类的概率。这个模型通过sigmoid函数输出一个介于0和1之间的值，表示样本属于正类（标签为1）的概率。决策边界定义在模型预测概率为0.5的位置。数学上，决策边界可以表示为：

\[ \sigma(\theta^T x) = 0.5 \]

其中，\( \sigma \) 是sigmoid函数，\( \theta \) 是参数向量（包括截距和斜率），\( x \) 是特征向量（通常包括一个常数项1来代表截距）。sigmoid函数的形式是：

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

当 \( \sigma(z) = 0.5 \) 时，\( z \) 必须为0（因为sigmoid函数在\( z=0 \)时输出0.5）。因此，决策边界对应于：

\[ \theta^T x = 0 \]

对于二维特征（如我们的案例，\( x_1 \) 和 \( x_2 \)），决策边界方程展开是：

\[ \theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0 \]

## 计算决策边界

在二维空间中，为了绘制决策边界，我们可以解这个方程相对于 \( x_2 \)，得到：

\[ x_2 = -\frac{\theta_1}{\theta_2} x_1 - \frac{\theta_0}{\theta_2} \]

这里：
- \( \theta_0 \) 是截距。
- \( \theta_1 \) 是\( x_1 \)的系数。
- \( \theta_2 \) 是\( x_2 \)的系数。

这个函数使用了上面提到的公式来计算 \( x_2 \)，从而在给定的 \( x_1 \) 值上绘制决策边界。

## 交叉熵损失函数
首先，定义逻辑回归使用的交叉熵损失函数：

\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]
\]

其中，\(\hat{p}^{(i)}\) 是模型预测样本 \(x^{(i)}\) 属于类别1的概率，且：

\[
\hat{p}^{(i)} = \sigma(\theta^T x^{(i)})
\]

\(\sigma(z)\) 是sigmoid函数：

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

### Sigmoid函数的导数
求sigmoid函数 \(\sigma(z)\) 的导数：

\[
\sigma'(z) = \frac{d}{dz} \left(\frac{1}{1 + e^{-z}}\right) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z))
\]

### 对数部分的导数
对于损失函数中的每一部分，应用对数求导规则：

\[
\frac{\partial}{\partial \theta_j} \log(\hat{p}^{(i)}) = \frac{1}{\hat{p}^{(i)}} \cdot \frac{\partial \hat{p}^{(i)}}{\partial \theta_j}
\]

\[
\frac{\partial}{\partial \theta_j} \log(1 - \hat{p}^{(i)}) = \frac{1}{1 - \hat{p}^{(i)}} \cdot \frac{\partial (1 - \hat{p}^{(i)})}{\partial \theta_j} = -\frac{1}{1 - \hat{p}^{(i)}} \cdot \frac{\partial \hat{p}^{(i)}}{\partial \theta_j}
\]

### 通过链式法则计算导数
将 \(\hat{p}^{(i)} = \sigma(\theta^T x^{(i)})\) 代入，求参数 \(\theta_j\) 的偏导数：

\[
\frac{\partial \hat{p}^{(i)}}{\partial \theta_j} = \sigma'(\theta^T x^{(i)}) \cdot \frac{\partial (\theta^T x^{(i)})}{\partial \theta_j}
\]

因为 \(\theta^T x^{(i)} = \sum_{k=1}^n \theta_k x_k^{(i)}\)，则：

\[
\frac{\partial (\theta^T x^{(i)})}{\partial \theta_j} = x_j^{(i)}
\]

所以：

\[
\frac{\partial \hat{p}^{(i)}}{\partial \theta_j} = \sigma(\theta^T x^{(i)}) (1 - \sigma(\theta^T x^{(i)})) x_j^{(i)} = \hat{p}^{(i)}(1 - \hat{p}^{(i)}) x_j^{(i)}
\]

### 损失函数的偏导数
代入损失函数中的对数部分，我们得到：

\[
\frac{\partial J(\theta)}{\partial \theta_j} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \frac{1}{\hat{p}^{(i)}} \hat{p}^{(i)}(1 - \hat{p}^{(i)}) x_j^{(i)} - (1 - y^{(i)}) \frac{1}{1 - \hat{p}^{(i)}} \hat{p}^{(i)}(1 - \hat{p}^{(i)}) x_j^{(i)} \right]
\]

简化这个表达式：

\[
\frac{\partial J(\theta)}{\partial \theta_j} = -\frac{1}{m} \sum_{i=1}^m \left[ (y^{(i)} - \hat{p}^{(i)}) x_j^{(i)} \right]
\]

这是逻辑回归中用于梯度下降算法的梯度表达式。通过这个梯度，我们可以更新参数 \(\theta\)，以优化模型性能。
