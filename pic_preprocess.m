function pic_preprocess = pic_preprocess(pic)
% ͼƬԤ�����Ӻ���
% ͼƬ��ɫ����
pic = 255 - pic;
% �趨��ֵ������ɫͼƬת�ɶ�ֵͼ��
pic = im2bw(pic, 0.4);
% �����������������ص�������� y �������� x
[y,x] = find(pic==1);
% ��ȡ�����������ֵ���С����
pic_preprocess = pic(min(y):max(y), min(x):max(x));
% ����ȡ�İ����������ֵ���С����ͼ��ת��16*16�ı�׼��ͼ��
pic_preprocess = imresize(pic_preprocess,[16,16]);
