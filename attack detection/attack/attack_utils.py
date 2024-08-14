import torch
from metric import ssim, psnr
from torch import nn
import random

def FGSM(net, X, y, e):

    x_compere = X.detach().clone()

    X.requires_grad_(True)
    pred = net(X)
    net.zero_grad()
    loss = nn.CrossEntropyLoss()(pred, y)
    loss.backward()
    X = X + e * torch.sign(X.grad)
    X = torch.clamp(X, 0, 1)

    SSIM = ssim(x_compere, X)
    PSNR = psnr(x_compere, X)
    return X, SSIM, PSNR


def DeepFool(net, device, X, y, max_iter):

    num_classes = 10
    overshoot = 0.02
    out = torch.zeros(X.shape).to(device)
    net.requires_grad_(False)
    x_compere = X.detach().clone()
    count = 0
    for x_, y_ in zip(X, y):
        x_ = x_.reshape((1, x_.shape[0], x_.shape[1], -1))
        w = torch.zeros(x_.shape)
        r_tot = torch.zeros(x_.shape)  # 记录累积扰动


        for epoch in range(max_iter):
            x_.requires_grad_(True)

            pred = net(x_)
            net.zero_grad()
            pred[0, y_].backward()
            if y_ != pred.argmax():
                break

            pert = torch.inf
            grad = x_.grad.detach().clone()

            for k in range(num_classes):
                if k == y_:
                    continue
                x_.grad.zero_()
                pred = net(x_)
                pred[0, k].backward()
                cur_grad = x_.grad.detach().clone()
                w_k = cur_grad - grad
                f_k = pred[0, k] - pred[0, y_]
                pert_k = abs(f_k) / torch.linalg.norm(w_k)
                x_.grad.zero_()
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r_i = (pert + 1e-8) * w / (torch.linalg.norm(w) + 1e-5)
            r_tot = r_tot.to(device) + r_i.to(device)
            with torch.no_grad():
                x_ = x_ + (1+overshoot) * r_tot
                # x_ = torch.clamp(x_, 0, 1)
        out[count] = x_
        count += 1
    SSIM = ssim(x_compere, out)
    PSNR = psnr(x_compere, out)
    return out, SSIM, PSNR


def BIM(net, X, e):
    x_compere = X.detach().clone()
    net.requires_grad_(False)

    for epoch in range(10):
        X.requires_grad_(True)
        pred = net(X)
        yll = torch.argmin(pred, dim=1)
        net.zero_grad()
        loss = nn.CrossEntropyLoss()(pred, yll)
        loss.backward()
        grad = X.grad.detach().clone()
        X.grad.zero_()
        with torch.no_grad():
            X = X - e * torch.sign(grad)
            X = torch.clamp(X, 0, 1)
    SSIM = ssim(x_compere, X)
    PSNR = psnr(x_compere, X)
    return X, SSIM, PSNR


def MJSMA(net, device, X, y, m):
    x_compere = X.detach().clone()
    out = torch.zeros(X.shape).to(device)
    count = 0
    for x_, y_ in zip(X, y):
        count = count + 1
        x_ = x_.reshape((1, x_.shape[0], x_.shape[1], -1))

        for epoch in range(m):
            x_.requires_grad_(True)
            pred = net(x_)
            net.zero_grad()

            if pred.argmax() != y_:
                break
            # 计算雅克比矩阵
            pred[0, y_].backward(retain_graph=True)
            jacobian = x_.grad.detach().clone()
            x_.grad.zero_()

            # 计算显著图
            adv_saliency_map = jacobian.reshape(x_.shape[1], x_.shape[2], x_.shape[3])

            # 计算显著图里的最大扰动
            mid_number = len(adv_saliency_map.reshape(-1)) // 2
            abs_vector = torch.abs(adv_saliency_map).reshape(-1)
            sort_vector, _ = torch.sort(abs_vector)

            mid_value = sort_vector[mid_number - 1]

            Mask_range = adv_saliency_map.gt(mid_value)
            adv_saliency_map = adv_saliency_map * Mask_range

            with torch.no_grad():
                x_ = torch.clamp(x_ - torch.sign(adv_saliency_map)*0.1, 0.0, 1.0)
        out[count - 1] = x_
    SSIM = ssim(x_compere, out)
    PSNR = psnr(x_compere, out)
    return out, SSIM, PSNR


def CW(net, device, X, y, max_iter):
    k = 40
    x_compere = X.detach().clone()
    out = torch.zeros(X.shape).to(device)
    confidence = 0.5
    num_classes = 10
    binary_search_steps = 5
    lower_bound = 0
    upper_bound = 1
    count = 0
    net.requires_grad_(False)

    for x_, y_ in zip(X, y):
        x_ = x_.reshape((1, x_.shape[0], x_.shape[1], -1))
        if y_ == 0:
            target = random.randint(1, 9)
        else:
            target = random.randint(0, y_-1)

        tlab = torch.eye(num_classes)[target].to(device)

        for outer_step in range(binary_search_steps):
            # 把原始图像转换成图像数据和扰动的形态
            modifier = torch.zeros_like(x_, dtype=torch.float).to(device)
            modifier.requires_grad = True

            # 定义优化器，仅优化modifer
            optimizer = torch.optim.Adam([modifier], lr=0.1)

            for iteration in range(1, max_iter + 1):
                optimizer.zero_grad()  # 梯度清零
                new_x = torch.tanh(modifier + x_.clone())
                output = net(new_x)

                if output.argmax() == target:
                    break

                # 定义cw中的损失函数
                # l2范数，用torch.dist()计算欧几里得距离，p=2为欧几里得距离，p=1为曼哈顿距离，即l1loss
                loss2 = torch.dist(new_x, torch.tanh(x_.clone()), p=2)
                real = torch.max(output * tlab)
                other = torch.max((1 - tlab) * output)
                loss1 = other - real + k
                loss1 = torch.clamp(loss1, min=0)  # 用clamp限制loss1最小为0

                loss = confidence * loss1 + loss2

                # 反向传播+梯度更新 使用retain_graph=True来保留计算图，以便下一次调用backward()方法。如果不设置retain_graph=True，则会在第一次反向传播后自动释放计算图。
                loss.backward(retain_graph=True)
                optimizer.step()

                o_bestscore = output.argmax()
                if (o_bestscore == target) and o_bestscore != -1:
                    upper_bound = min(upper_bound, confidence)
                    confidence = (lower_bound + upper_bound) / 2
                else:
                    lower_bound = max(lower_bound, confidence)
                    confidence = (lower_bound + upper_bound) / 2

        out[count] = new_x
        count = count + 1
    SSIM = ssim(x_compere, out)
    PSNR = psnr(x_compere, out)
    return out, SSIM, PSNR
