import torch
import torch.nn as nn

class BasicBlock(nn.Module):
	"""
	Basic block 
	"""

	def __init__(self, inplanes, planes, stride=1, dilation=1):
		super(SEBasicBlock, self).__init__()
		
		# if change in number of filters
		if inplanes != planes: 
			self.shortcut = nn.Sequential(
				nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
			)

		# original resolution path (original branches)
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, x, residual=None):
		if residual is None:
			residual = self.shortcut(x) if hasattr(self, "shortcut") else x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	"""
	Bottleneck block of ResNeXt architectures[1].
	Dynamic scaling policy (DSP) is based on the elastic module[2].

	Reference:
	[1] Xie, Saining, et al. 
	"Aggregated residual transformations for deep neural networks." CVPR. 2017.
	[2] Wang, Huiyu, et al. 
	"Elastic: Improving cnns with dynamic scaling policies." CVPR. 2019.
	"""

	def __init__(self, inplanes, planes, stride=1, dilation=1):
		super(Bottleneck, self).__init__()
		
		bottel_plane = planes // 2
		cardinality = bottel_plane // 4
		
				
		# if change in number of filters
		if inplanes != planes: 
			self.shortcut = nn.Sequential(
				nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
			)

		# original resolution path (original branches)
		self.conv1 = nn.Conv2d(inplanes, bottel_plane,
							   kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(bottel_plane)
		self.conv2 = nn.Conv2d(bottel_plane, bottel_plane, kernel_size=3,
							   stride=stride, padding=dilation, bias=False,
							   dilation=dilation, groups=cardinality)
		self.bn2 = nn.BatchNorm2d(bottel_plane)
		self.conv3 = nn.Conv2d(bottel_plane, planes,
							   kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, x, residual=None):
		if residual is None:
			residual = self.shortcut(x) if hasattr(self, "shortcut") else x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		
		out = self.conv3(out)
		out = self.bn3(out)

		out += residual
		out = self.relu(out)

		return out


class SEBasicBlock(nn.Module):
	def __init__(self, inplanes, planes, stride=1, dilation=1, reduction=8):
		super(SEBasicBlock, self).__init__()

		# if change in number of filters
		if inplanes != planes: 
			self.shortcut = nn.Sequential(
				nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
			)

		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.se = SELayer(planes, reduction)
		self.stride = stride

	def forward(self, x, residual=None):
		if residual is None:
			residual = self.shortcut(x) if hasattr(self, "shortcut") else x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.se(out) + residual
		out = self.relu(out)

		return out


class SEBottleneck(nn.Module):
	def __init__(self, inplanes, planes, stride=1, dilation=1, reduction=8):
		super(SEBottleneck, self).__init__()

		bottel_plane = planes // 4
		cardinality = bottel_plane // 4

		# if change in number of filters
		if inplanes != planes: 
			self.shortcut = nn.Sequential(
				nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
			)
		
		self.conv1 = nn.Conv2d(inplanes, bottel_plane, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(bottel_plane)
		self.conv2 = nn.Conv2d(bottel_plane, bottel_plane, kernel_size=3, stride=stride,
							   padding=dilation, bias=False, dilation=dilation, groups=cardinality)
		self.bn2 = nn.BatchNorm2d(bottel_plane)
		self.conv3 = nn.Conv2d(bottel_plane, planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.se = SELayer(planes, reduction)

	def forward(self, x, residual=None):
		if residual is None:
			residual = self.shortcut(x) if hasattr(self, "shortcut") else x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		out = self.se(out) + residual
		out = self.relu(out)

		return out


class SELayer(nn.Module):
	def __init__(self, channel, reduction=8):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
				nn.Linear(channel, channel // reduction),
				nn.ReLU(inplace=True),
				nn.Linear(channel // reduction, channel),
				nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y
