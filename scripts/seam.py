import numpy as np
import cv2
import os.path
from skimage.feature import hog
from skimage import exposure
import cPickle as cpk
import time
from matplotlib import pyplot as plt

IMG         = 'img1'
IMG_NAME    = '../img/' + IMG + '.jpg'
SAVE_DIR    = '../img/'
SEAMHE_FILE = '../data/seamHE'
SEAMVE_FILE = '../data/seamVE'
TRACKV_FILE = '../data/vTrack'
TRACKH_FILE = '../data/hTrack'

class Resizer:

	def __init__(self, imgFileName):
		if type(imgFileName) is str:
			self.raw = cv2.imread(imgFileName)
		else:
			self.raw = imgFileName
		self.raw = self.initTestImg()
		[self.n, self.m, self.channels] = self.raw.shape
		self.k = 5
		self.energy = []
		self.seamVE = []
		self.seamHE = []
		self.vTrack = []
		self.hTrack = []
		self.vSeam  = []
		self.hSeam  = []

	def initTestImg(self):
		tmp = (np.random.rand(15, 15, 3) * 256).astype(np.uint8)
		t = [0, 0, 0]
		for i in range(tmp.shape[0]):
			tmp[i, 3] = t
			tmp[i, 4]   = t
			if i > tmp.shape[0]-1:
				tmp[i, 5] = t
		return tmp
	
	def deltaX(self, avg):
		return cv2.Sobel(self.raw.mean(axis=2), cv2.CV_8U, 1, 0, ksize=self.k)		
	def deltaY(self, avg):
		return cv2.Sobel(self.raw.mean(axis=2), cv2.CV_8U, 0, 1, ksize=self.k)

	def show(self, winName='image', img=None, save=False):
		if img is None:
			print 'disp raw'
			img = self.raw
		cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)
		cv2.imshow(winName, img)
		if save == True:
			cv2.imwrite(SAVE_DIR + winName, img)
	@staticmethod
	def normalize(img, l=0, r=255):
		rangeNew = r-l+1
		minV = np.min(img)
		maxV = np.max(img)
		rangeOld = maxV-minV+1
		scale = rangeNew/rangeOld
		img = np.floor(l + (img-minV) * scale)
		return img.astype(np.uint8) 
	def makeEnergyMap(self, avg):
		self.energy = np.abs(self.deltaX(avg)) + np.abs(self.deltaY(avg))
		
	def getSeamE(self, x, y, seam, avg):
		if avg == True:
			axis = None
		else:
			axis = 2
		try:
			if seam == 'v':
				if y == 0:
					self.vTrack[x, y] = y + np.argmin([self.seamVE[x-1, y], self.seamVE[x-1, y+1]])
				elif y == self.m-1:
					self.vTrack[x, y] = y + np.argmin([self.seamVE[x-1, y-1], self.seamVE[x-1, y]]) - 1
				else:
					self.vTrack[x, y] = y + np.argmin([self.seamVE[x-1, y-1], self.seamVE[x-1, y], self.seamVE[x-1, y+1]], axis=axis) - 1
				return self.energy[x, y] + self.seamVE[x-1, (self.vTrack[x, y])]
			else:
				if y == 0:
					self.hTrack[y, x] = y + np.argmin([self.seamHE[y, x-1], self.seamHE[y+1, x-1]])
				elif y == self.n-1:
					self.hTrack[y, x]= y + np.argmin([self.seamHE[y-1, x-1], self.seamHE[y, x-1]]) - 1
				else:
					self.hTrack[y, x] = y + np.argmin([self.seamHE[y-1, x-1], self.seamHE[y, x-1], self.seamHE[y+1, x-1]], axis=axis) - 1
				return self.energy[y, x] + self.seamHE[(self.hTrack[y, x]), x-1]
		except Exception as e:
			print 'error:', seam, x, y
			print e
			raise

	def pxInd(self, x, y, seam, avg):
		if seam == 'h':
			if avg == True:
				seamE = self.seamHE.transpose(1, 0)
			else:
				seamE = self.seamHE.transpose(1, 0, 2)
		else:
			if avg == True:
				seamE = self.seamVE.transpose(1, 0)
			else:
				seamE = self.seamVE.transpose(1, 0, 2)		
			
		l = y-1
		c = y
		r = y+1

	def popSeamE(self, seam, avg):
		if seam == 'h':
			#if os.path.isfile(SEAMHE_FILE):
			#	self.seamHE = cpk.load(open(SEAMHE_FILE, 'rb'))
			#	self.hTrack = cpk.load(open(TRACKH_FILE, 'rb'))
			#	print 'seamHE read from', SEAMHE_FILE
			#	return
			self.seamHE = self.seamHE.astype(np.float64)
			if avg == True:
				seamE = self.seamHE.transpose(1, 0)
			else:
				seamE = self.seamHE.transpose(1, 0, 2)
		else:
			#if os.path.isfile(SEAMVE_FILE):
			#	self.seamVE = cpk.load(open(SEAMVE_FILE, 'rb'))
			#	self.vTrack = cpk.load(open(TRACKV_FILE, 'rb'))
			#	print 'seamVE read from', SEAMVE_FILE
			#	return
			self.seamVE = self.seamVE.astype(np.float64)
			seamE = self.seamVE

		for i in range(seamE.shape[0]-1):
			for j in range(seamE.shape[1]):
				seamE[i+1, j] = self.getSeamE(i+1, j, seam, avg)

		if seam == 'h':
			self.seamHE = Resizer.normalize(self.seamHE).astype(np.uint8)
			cpk.dump(self.seamHE, open(SEAMHE_FILE, 'wb'))
			cpk.dump(self.hTrack, open(TRACKH_FILE, 'wb'))
			print 'seamHE written to', SEAMHE_FILE
		else:
			self.seamVE = Resizer.normalize(self.seamVE).astype(np.uint8)
			cpk.dump(self.seamVE, open(SEAMVE_FILE, 'wb'))
			cpk.dump(self.vTrack, open(TRACKV_FILE, 'wb'))
			print 'seamVE written to', SEAMVE_FILE

	def makeSeamEMap(self, avg=True):
		self.makeEnergyMap(avg)
		
		self.vTrack = np.zeros(self.energy.shape)
		self.hTrack = np.zeros(self.energy.shape)
		self.seamVE = self.energy.copy()
		self.seamHE = self.energy.copy()
		
		self.vTrack[0]    = np.arange(self.vTrack.shape[1])
		self.hTrack[:, 0] = np.arange(self.hTrack.shape[0])
		
		self.popSeamE(seam='v', avg=avg)
		self.popSeamE(seam='h', avg=avg)
		
	def getMinKSeams(self, k, seamOri, avg=True):
		if seamOri == 'h':
			if avg == True:
				seamE = self.seamHE.transpose(1, 0)
				track = self.hTrack.transpose(1, 0)
			else:
				seamE = self.seamHE.transpose(1, 0, 2)		
				track = self.hTrack.transpose(1, 0, 2)
		else:
			seamE = self.seamVE
			track = self.vTrack
		if avg == True:
			axis = None
		else:
			axis = 2
		seam = np.argsort(seamE[-1], axis=0)[:k]
		seamStore = np.zeros([seamE.shape[0], seam.shape[0]])
		## init seamStore for avg==False also
		
		seamStore[-1, :] = seam
		for i in range(seamStore.shape[1]):
			for j in range(seamStore.shape[0]-1):
				seamStore[-2-j, i] = track[-1-j, seamStore[-1-j, i]]
		
		#print seamStore
		if seamOri == 'h':
			if avg == True:
				self.hSeam = seamStore.transpose(1, 0)
			else:
				self.hSeam = seamStore.transpose(1, 0, 2)
		else:
			self.vSeam = seamStore

	def overlaySeams(self, k, seam, avg=True):
		if seam == 'h':
			if avg == True:
				seamStore = self.hSeam.transpose(1, 0)		
			else:
				seamStore = self.hSeam.transpose(1, 0, 2)	
			k = np.minimum(k, seamStore.shape[0])
			raw = self.raw.transpose(1, 0, 2).copy()
		else:
			seamStore = self.vSeam
			k = np.minimum(k, seamStore.shape[1])
			raw = self.raw.copy()

		raw = np.ones(raw.shape) * 100
		tt  = np.zeros(raw.shape)
		tt  = tt[:, :, 1]
		for i in range(seamStore.shape[1]):
			for j in range(seamStore.shape[0]):
				raw[j, seamStore[j, i]] = [0, 255, 0]
				tt[j, seamStore[j, i]]  = 11
		print tt.astype(np.int)
				
		if seam == 'h':
			return raw.transpose(1, 0, 2)		
		return raw
	
	def removeSeams(self, k, seam, avg=True):
		if seam == 'h':
			if avg == True:
				seamStore = self.hSeam.transpose(1, 0)		
			else:
				seamStore = self.hSeam.transpose(1, 0, 2)	
			k = np.minimum(k, seamStore.shape[0])
			raw = self.raw.transpose(1, 0, 2).copy()
		else:
			seamStore = self.vSeam
			k = np.minimum(k, seamStore.shape[1])
			raw = self.raw.copy()

		[rows, cols, channels] = raw.shape
		imgSize = rows * cols

		#raw = raw.ravel(order='F')
		toDelete = []
		
		#skip = 0
		copyImg = np.zeros([rows, cols-k, channels])
		rawC = raw.copy()
		raw = np.zeros([raw.shape[0], raw.shape[1]-k, raw.shape[3]])
		kk   = 0
		for i in range(rawC.shape[0]):
			
			raw[i, seamStore[i, kj]:] = raw[i, seamStore[i, kj]+1:]
			for j in range(rawC.shape[1]):
				if not np.any(seamStore[i] == j):
					raw[i, kj] = rawC[i, j]
					kj += 1
		print raw.shape			
		
		for i in range(seamStore.shape[1]):
		#	print i, seamStore[-1, i]
			for j in range(seamStore.shape[0]):
				start = int(seamStore[j, i])-0
				#if j==0:
				#	print type(seamStore[j, i]), seamStore[j, i]
				for col in range(cols-start-1):
					raw[j, start+col] = raw[j, start+col+1]
		
		#raw = rawC#[:, :-k, :]
		
		#raw = np.delete(raw, toDelete)
		#raw = raw.reshape(rows, cols-k, channels)
		#raw = raw[:, :-k, :]
		print raw.shape
		if seam == 'h':
			raw = raw.transpose(1, 0, 2)	

		return raw.astype(np.uint8)
		
	def overlaySeams2(self, k, seam, avg=True):
		if seam == 'h':
			seamE = self.seamHE.transpose(1, 0).copy().astype(np.int)
		else:
			seamE = self.seamVE.copy().astype(np.int)

		for m in range(k):
			minCol = np.argmin(seamE[-1])	
			for i in range(seamE.shape[0]):
				for j in range(seamE.shape[1]):
					if j == minCol:
						#print seamE.shape[0]-i-1, j,
						seamE[-i-1, j] = 400
				if i < seamE.shape[0]-1:
					minVal = seamE[-i-2, j]
					if minCol > 0 and minCol < seamE.shape[1]-1:
						minCol += np.argmin([seamE[-i-2, minCol-1], seamE[-i-2, minCol], seamE[-i-2, minCol+1]]) - 1
					elif minCol == seamE.shape[1]-1:
						minCol += np.argmin([seamE[-i-2, minCol-1], seamE[-i-2, minCol]]) + 1
					elif minCol == 0:
						minCol += np.argmin([seamE[-i-2, minCol], seamE[-i-2, minCol+1]]) + 1
		#print 'o2'
		print
		tmp = seamE.copy()
		tmp[tmp != 400] = 0
		tmp[tmp == 400] = 22
		#print tmp

		seamE[seamE==400] = 0
		seamE = seamE.astype(np.uint8)
		if seam == 'h':
			return seamE.transpose(1, 0)
		return seamE

	def removeSeams2(self, k, seam, avg=True):
		if seam == 'h':
			seamE = self.seamHE.transpose(1, 0).copy()
			img   = np.zeros(self.raw.transpose(1, 0, 2).shape)
			raw   = self.raw.transpose(1, 0, 2)
		else:
			seamE = self.seamVE.copy()
			img   = np.zeros(self.raw.shape)
			img   = self.raw
		#print img.shape, self.raw.shape, np.mean(img), np.mean(self.raw)
		seamE = seamE.astype(np.int)
		for m in range(k):
			minCol = np.argmin(seamE[-1])	
			seamE[-1, minCol] = 1000
			#print m, minCol
			for i in range(img.shape[0]-1):
				pastMinCol = False
				for j in range(img.shape[1]-k):
					if j == minCol:
						pastMinCol = True
						#print k, i, j
					try:
						if pastMinCol == True:
							img[-i-1, j] = img[-i-1, j+1]
						else:
							img[-i-1, j] = img[-i-1, j]
					except:
						print k, i, j
						raise
				if i < seamE.shape[0]-1:
					minVal = seamE[-i-2, j]
					if minCol > 0 and minCol < seamE.shape[1]-1:
						minCol += np.argmin([seamE[-i-2, minCol-1], seamE[-i-2, minCol], seamE[-i-2, minCol+1]]) - 1
					elif minCol == seamE.shape[1]-1:
						minCol += np.argmin([seamE[-i-2, minCol-1], seamE[-i-2, minCol]]) + 1
					elif minCol == 0:
						minCol += np.argmin([seamE[-i-2, minCol], seamE[-i-2, minCol+1]]) + 1
	
		print type(img), np.min(img), np.max(img), np.mean(img), np.mean(self.raw)

		if seam == 'h':
			return img.transpose(1, 0, 2).astype(np.uint8)
		return img.astype(np.uint8)

	'''def hoG(self):
		fd, hog_image = hog(self.raw.mean(axis=(2)), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)		
		hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
		print hog_image.shape, np.min(hog_image), np.max(hog_image)
		print hog_image
		self.show(img=hog_image)
	'''
if __name__=='__main__':
	r = Resizer(IMG_NAME)
	#r.hoG()
	r.makeSeamEMap()
	#r.show(winName='energy', img=r.seamVE)
	t1 = time.clock()	
	r.getMinKSeams(k=10, seamOri='v')
	r.overlaySeams(k=10, seam='v')
	#t2 = time.clock()
	#r.show(winName=IMG + '-L1-a.jpg', img=r.removeSeams(k=100, seam='v'), save=False)
	#t3 = time.clock()
	#print 'mink', (t2-t1)*1000
	#print 'o1', (t3-t2) * 1000, (t3-t1) * 1000
	#r.show(winName='seamH', img=r.seamHE);
	#r.show()
	#r.getMinKSeams(k=100, seamOri='v')
	#img = r.removeSeams(k=100, seam='v')
	#r.show(winName='cropv', img=img)
	#cv2.imwrite('img1.jpg', img)
	'''r = Resizer(img.copy())
	r.makeSeamEMap()
	r.getMinKSeams(k=200, seamOri='h')
	img = r.removeSeams(k=200, seam='h')
	r.show(winName='cropvh', img=img)
	#r.show(winName='seamOverlay', img=r.overlaySeams(k=2, seam='v'))
	#r.makeEnergyMap(True)
	#r.vTrack = np.zeros(r.energy.shape)
	#r.seamVE = r.energy.copy()
	#print r.getSeamE(210, 100, seam='v', avg=True)

	#print np.max(r.deltaY(avg=True))
	#print np.min(r.deltaY(avg=True))
	#for i in range(r.seamVE.shape[2]):
	#r.show(winName='img_' + str(i), img=r.seamVE[:, :, i])	
	'''
	cv2.waitKey(0)
	

'''
move inside Resizer





'''
