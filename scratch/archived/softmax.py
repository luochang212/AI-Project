# %%
import torch
from IPython import display
from d2l import torch as d2l


class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class Softmax:

    def __init__(self, batch_size, num_inputs, num_outputs):
        self.batch_size = batch_size
        self.W = torch.normal(0, 0.01
                         , size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def softmax(self, X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition
    
    def net(self, X):
        return self.softmax(torch.matmul(X.reshape((-1, self.W.shape[0])), self.W) + self.b)

    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])
    
    @staticmethod
    def sgd(params, lr, batch_size):
        """Minibatch stochastic gradient descent.

        Defined in :numref:`sec_utils`"""
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    def updater(self, batch_size, lr=0.1):
        return self.sgd([self.W, self.b], lr, batch_size)

    def accuracy(self, y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def evaluate_accuracy(self, net, data_iter):
        """计算在指定数据集上模型的精度"""
        metric = Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(self.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    def train_epoch(self, train_iter, net, loss, updater):
        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, net, train_iter, test_iter, loss, num_epochs, updater):
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_iter, net, loss, updater)
            test_acc = self.evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        print(train_loss, train_acc)
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc
    
    def main(self):
        train_iter, test_iter = d2l.load_data_fashion_mnist(self.batch_size)
        num_epochs = 10
        self.train(net=self.net, train_iter=train_iter, test_iter=test_iter, loss=self.cross_entropy, num_epochs=num_epochs, updater=self.updater)


if __name__ == '__main__':
    s = Softmax(batch_size=256, num_inputs=784, num_outputs=10)
    s.main()

# %%
