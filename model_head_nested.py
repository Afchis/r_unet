import torch
import torch.nn as nn

from args import *
from model_paths import *


'''
Model head
'''
class UNetDesigner(nn.Module):    
	def __init__(self, d1, d2, d3, d4, b_, u4, u3, u2, u1,
				 input_size=INPUT_SIZE, input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES):
		super(UNetDesigner, self).__init__()
		self.num_classes = NUM_CLASSES
		self.d1, self.d2, self.d3, self.d4, self.b, self.u1, self.u2, self.u3, self.u4 = d1, d2, d3, d4, b_, u1, u2, u3, u4
		self.input_size = input_size
		self.input_chennels = input_channels
		self.ch_list = [self.input_chennels, 64, 128, 256, 512, 1024]
		self.input_x2 = int(self.input_size / 2)
		self.input_x4 = int(self.input_size / 4)
		self.input_x8 = int(self.input_size / 8)
		self.input_x16 = int(self.input_size / 16)

		 ##### Down_1 layer ##### input_size = 256                                      # Channels
		if self.d1 == 'False':
			self.down1 = nn.Sequential(ConvRelu(self.ch_list[0], self.ch_list[1]),
									   ConvRelu(self.ch_list[1], self.ch_list[1])
									   )
																						# 3  -->64
		else:                                                                           # 64 -->64
			self.down1 = nn.Sequential(ConvRnnRelu(self.ch_list[0], self.ch_list[1],    
												   self.input_size, self.d1),               
									   ConvRelu(self.ch_list[1], self.ch_list[1])                   
									   )
		self.down1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		 ##### Down_2 layer ##### input_size = 128
		if self.d2 == 'False':
			self.down2 = nn.Sequential(ConvRelu(self.ch_list[1], self.ch_list[2]),
									   ConvRelu(self.ch_list[2], self.ch_list[2])
									   )
																						# 64 -->128
		else:                                                                           # 128-->128
			self.down2 = nn.Sequential(ConvRnnRelu(self.ch_list[1], self.ch_list[2],
												   self.input_x2, self.d2),
									   ConvRelu(self.ch_list[2], self.ch_list[2])
																			 )
		self.down2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		 ##### Down_3 layer ##### input_size = 64
		if self.d3 == 'False':
			self.down3 = nn.Sequential(ConvRelu(self.ch_list[2], self.ch_list[3]),
				                       ConvRelu(self.ch_list[3], self.ch_list[3])
								       )
																			            # 128-->256
		else:                                                                           # 256-->256
			self.down3 = nn.Sequential(ConvRnnRelu(self.ch_list[2], self.ch_list[3], 
												   self.input_x4, self.d3),
									   ConvRelu(self.ch_list[3], self.ch_list[3])
									   )
		self.down3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		 ##### Down_4 layer ##### input_size = 32
		if self.d4 == 'False':
			self.down4 = nn.Sequential(ConvRelu(self.ch_list[3], self.ch_list[4]),
				                       ConvRelu(self.ch_list[4], self.ch_list[4])
								       )
																			            # 256-->512
		else:                                                                           # 512-->512
			self.down4 = nn.Sequential(ConvRnnRelu(self.ch_list[3], self.ch_list[4], 
												   self.input_x8, self.d4),
									   ConvRelu(self.ch_list[4], self.ch_list[4])
									   )
		self.down4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		 ##### Bottom layer ##### input_size = 16
		if self.b == 'False':
			self.bottom = nn.Sequential(ConvRelu(self.ch_list[4], self.ch_list[5]),
										ConvRelu(self.ch_list[5], self.ch_list[5])
										)
																						# 512-->1028
		else:                                                                           # 1028-->1028
			self.bottom = nn.Sequential(ConvRnnRelu(self.ch_list[4], self.ch_list[5], 
													self.input_x16, self.b),
										ConvRelu(self.ch_list[5], self.ch_list[5])
										)

		 ##### Up_4 layer #####
		self.up_cat_4 = UpAndCat()
		if self.u4 == 'False':
			self.up_conv_4 = nn.Sequential(ConvRelu(self.ch_list[5]+self.ch_list[4], 
													self.ch_list[4]),
								           ConvRelu(self.ch_list[4], self.ch_list[4])
								           )
																						# 1540-->512
		else:                                                                           # 512-->512
			self.up_conv_4 = nn.Sequential(ConvRnnRelu(self.ch_list[5]+self.ch_list[4], 
													   self.ch_list[4], 
													   self.input_x8, self.u3),
										   ConvRelu(self.ch_list[4], self.ch_list[4])
										   ) 

		 ##### Up_3 layer #####
		self.up_cat_3 = UpAndCat_2()
		if self.u3 == 'False':
			self.up_conv_3 = nn.Sequential(ConvRelu(1024, 
													self.ch_list[3]),
								           ConvRelu(self.ch_list[3], self.ch_list[3])
								           )
																						# 768-->256
		else:                                                                           # 256-->256
			self.up_conv_3 = nn.Sequential(ConvRnnRelu(self.ch_list[4]+self.ch_list[3], 
													   self.ch_list[3], 
													   self.input_x4, self.u3),
										   ConvRelu(self.ch_list[3], self.ch_list[3])
										   )   

		 ##### Up_2 layer #####
		self.up_cat_2 = UpAndCat_3()
		if self.u2 == 'False':
			self.up_conv_2 = nn.Sequential(ConvRelu(640, 
													self.ch_list[2]),
										   ConvRelu(self.ch_list[2], self.ch_list[2])
										   )
																						# 394-->128
		else:                                                                           # 128-->128
			self.up_conv_2 = nn.Sequential(ConvRnnRelu(self.ch_list[3]+self.ch_list[2], 
													   self.ch_list[2], 
													   self.input_x2, self.u2),
									       ConvRelu(self.ch_list[2], self.ch_list[2])
										   )

		 ##### Up_1 layer #####
		self.up_cat_1 = UpAndCat_4()
		if self.u1 == 'False':
			self.up_conv_1 = nn.Sequential(ConvRelu(384, 
													self.ch_list[1]),
										   ConvRelu(self.ch_list[1], self.ch_list[1])
										   )
																						# 128-->64
		else:                                                                           # 64 -->64
			self.up_conv_1 = nn.Sequential(ConvRnnRelu(self.ch_list[2]+self.ch_list[1], 
													   self.ch_list[1], 
													   self.input_size, self.u1),
										   ConvRelu(self.ch_list[1], self.ch_list[1])
										   )

	     ##### Final layer #####
		self.final = nn.Sequential(nn.Conv2d(self.ch_list[1], self.num_classes, kernel_size=1),
								   # nn.Softmax()								 
								   )                                                    # 64-->NUM_CLASSES

		self.o_cat_1_1 = UpAndCat()
		self.o_conv_1_1 = nn.Sequential(ConvRelu(self.ch_list[1]+self.ch_list[1], 
                                             	 self.ch_list[1]),
										ConvRelu(self.ch_list[1], self.ch_list[1])
                                    	)

		self.o_cat_1_2 = UpAndCat_2()
		self.o_conv_1_2 = nn.Sequential(ConvRelu(256, 
                                             self.ch_list[1]),
                                    ConvRelu(self.ch_list[1], self.ch_list[1])
                                    )

		self.o_cat_1_3 = UpAndCat_3()
		self.o_conv_1_3 = nn.Sequential(ConvRelu(320, 
                                             self.ch_list[1]),
                                    ConvRelu(self.ch_list[1], self.ch_list[1])
                                    )
		
		self.o_cat_2_1 = UpAndCat()
		self.o_conv_2_1 = nn.Sequential(ConvRelu(self.ch_list[2]+self.ch_list[2], 
                                             self.ch_list[2]),
                                    ConvRelu(self.ch_list[2], self.ch_list[2])
                                    )

		self.o_cat_2_2 = UpAndCat_2()
		self.o_conv_2_2 = nn.Sequential(ConvRelu(512, 
                                             self.ch_list[2]),
                                    ConvRelu(self.ch_list[2], self.ch_list[2])
                                    )

		self.o_cat_3_1 = UpAndCat()
		self.o_conv_3_1 = nn.Sequential(ConvRelu(self.ch_list[3]+self.ch_list[3], 
                                             self.ch_list[3]),
                                    ConvRelu(self.ch_list[3], self.ch_list[3])
                                    )


	def forward(self, x):
		x = x.reshape(-1, self.input_chennels, self.input_size, self.input_size) 

		down1_feat = self.down1(x)
		pool1 = self.down1_pool(down1_feat)

		o1_1 = self.o_cat_1_1(pool1, down1_feat)
		o1_1 = self.o_conv_1_1(o1_1)

		down2_feat = self.down2(pool1)
		pool2 = self.down2_pool(down2_feat)

		o2_1 = self.o_cat_2_1(pool2, down2_feat)
		o2_1 = self.o_conv_2_1(o2_1)

		o1_2 = self.o_cat_1_2(o2_1, o1_1, down1_feat)
		o1_2 = self.o_conv_1_2(o1_2)

		down3_feat = self.down3(pool2)
		pool3 = self.down3_pool(down3_feat)
		
		o3_1 = self.o_cat_3_1(pool3, down3_feat)
		o3_1 = self.o_conv_3_1(o3_1)

		o2_2 = self.o_cat_2_2(o3_1, o2_1, down2_feat)
		o2_2 = self.o_conv_2_2(o2_2)

		o1_3 = self.o_cat_1_3(o2_2, o1_2, o1_1, down1_feat)
		o1_3 = self.o_conv_1_3(o1_3)


		down4_feat = self.down4(pool3)
		pool4 = self.down4_pool(down4_feat)

		bottom_feat = self.bottom(pool4)

		up_feat4 = self.up_cat_4(bottom_feat, down4_feat)
		up_feat4 = self.up_conv_4(up_feat4)

		up_feat3 = self.up_cat_3(up_feat4, o3_1 ,down3_feat)
		up_feat3 = self.up_conv_3(up_feat3)
		
		up_feat2 = self.up_cat_2(up_feat3, o2_1, o2_2, down2_feat)
		up_feat2 = self.up_conv_2(up_feat2)
		
		up_feat1 = self.up_cat_1(up_feat2, o1_1, o1_2, o1_3, down1_feat)
		up_feat1 = self.up_conv_1(up_feat1)
		
		out = self.final(up_feat1)

		if __name__ == '__main__':
			print('x', x.shape)
			print('pool1', pool1.shape)
			print('pool2', pool2.shape)
			print('pool3', pool3.shape)
			print('pool4', pool4.shape)
			print('bottom_feat', bottom_feat.shape)
			print('up_feat4', up_feat4.shape)
			print('up_feat3', up_feat3.shape)
			print('up_feat2', up_feat2.shape)
			print('up_feat1', up_feat1.shape)
			print('out', out.shape)

		return out

if __name__ == '__main__':
	import torch
	tensor = torch.rand([1, 1, 3, 256, 256])
	model = UNetDesigner('False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', )
	model(tensor)