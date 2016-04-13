import numpy as np
import cv2
import os.path
from skimage.feature import hog
from skimage import exposure
import cPickle as cpk
import time
import logging
from matplotlib import pyplot as plt

IMG_ARR     = ['img1', 'img2', 'img3', 'img4']
#IMG_ARR     = ['img1']
SEAM        = 'v'
SEAM_K      = 100
#IMG_NAME    = '../img/' + IMG + '.jpg'
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
		#self.raw = self.initTestImg()
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
			#print 'disp raw'
			img = self.raw
		#cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)
		#cv2.imshow(winName, img)
		if save == True:
			cv2.imwrite(SAVE_DIR + winName, img)
			logging.info('Image saved at %s%s', SAVE_DIR, winName)
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
			print 'error:', seam, x, y, type(self.vTrack), self.vTrack.shape
			print e
			raise

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
		#	cpk.dump(self.seamHE, open(SEAMHE_FILE, 'wb'))
		#	cpk.dump(self.hTrack, open(TRACKH_FILE, 'wb'))
		#	print 'seamHE written to', SEAMHE_FILE
		else:
			self.seamVE = Resizer.normalize(self.seamVE).astype(np.uint8)
		#	cpk.dump(self.seamVE, open(SEAMVE_FILE, 'wb'))
		#	cpk.dump(self.vTrack, open(TRACKV_FILE, 'wb'))
		#	print 'seamVE written to', SEAMVE_FILE

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
		
	def getMinSeam(self, seamOri, avg=True):
		self.makeEnergyMap(avg)
		self.seamHE = self.energy.copy()
		self.seamVE = self.energy.copy()
		self.vTrack = np.zeros(self.energy.shape)
		self.hTrack = np.zeros(self.energy.shape)

		self.vTrack[0]    = np.arange(self.vTrack.shape[1])
		self.hTrack[:, 0] = np.arange(self.hTrack.shape[0])
		self.popSeamE(seam=seamOri, avg=avg)
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
		seam = np.argsort(seamE[-1], axis=0)[:1]
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
		#print tt.astype(np.int)
				
		if seam == 'h':
			return raw.transpose(1, 0, 2)		
		return raw

	def addSeams(self, k, seam, avg=True):
		logging.info('Adding %d %s seams', k, seam)
		t0 = time.clock()
		for s in range(k):
			t1 = time.clock()
			self.getMinSeam(seamOri=seam, avg=avg)
			if seam == 'h':
				if avg == True:
					seamStore = self.hSeam.transpose(1, 0)		
				else:
					seamStore = self.hSeam.transpose(1, 0, 2)	
				rawC = self.raw.transpose(1, 0, 2)
			else:
				seamStore = self.vSeam
				rawC = self.raw
			
			raw = np.zeros([rawC.shape[0], rawC.shape[1]+1, rawC.shape[2]])
			for i in range(seamStore.shape[0]):
				raw[i, :seamStore[i, 0]]   = rawC[i, :seamStore[i, 0]]
				raw[i, seamStore[i, 0]]    = 0.5* rawC[i, seamStore[i, 0]-1] + 0.5 * rawC[i, seamStore[i, 0]]
				raw[i, seamStore[i, 0]+1]    = 0.5* rawC[i, seamStore[i, 0]+1] + 0.5 * rawC[i, seamStore[i, 0]]
				raw[i, seamStore[i, 0]+2:] = rawC[i, seamStore[i, 0]+1:]
					
			#raw = raw[:, :-1, :]
			if seam == 'h':
				self.raw = raw.transpose(1, 0, 2).astype(np.uint8)	
			else:
				self.raw = raw.astype(np.uint8)
			[self.n, self.m] = self.raw.shape[:2]
			t2 = time.clock()
			print 'Added seam:', s, 'in', (t2-t1) #, np.max(self.raw), np.min(self.raw), type(self.raw)
		tf = time.clock()
		logging.info('Added  %d %s seams in %f s %s', k, seam, (t0-tf), self.raw.shape)
		return raw.astype(np.uint8)
	
	def removeSeams(self, k, seam, avg=True):
		logging.info('Removing %d %s seams', k, seam)
		t0 = time.clock()
		for s in range(k):
			t1 = time.clock()
			self.getMinSeam(seamOri=seam, avg=avg)
			if seam == 'h':
				if avg == True:
					seamStore = self.hSeam.transpose(1, 0)		
				else:
					seamStore = self.hSeam.transpose(1, 0, 2)	
				raw = self.raw.transpose(1, 0, 2)
			else:
				seamStore = self.vSeam
				raw = self.raw
			
			for i in range(seamStore.shape[0]):
				raw[i, seamStore[i, 0]:-1] = raw[i, seamStore[i, 0]+1:]
					
			raw = raw[:, :-1, :]
			if seam == 'h':
				self.raw = raw.transpose(1, 0, 2)	
			else:
				self.raw = raw
			[self.n, self.m] = self.raw.shape[:2]
			t2 = time.clock()
			print 'Removed seam:', s, 'in', (t2-t1)
		tf = time.clock()
		logging.info('Removed  %d %s seams in %f s %s', k, seam, (t0-tf), self.raw.shape)
		return raw.astype(np.uint8)
		
if __name__=='__main__':
	logging.basicConfig(filename='Resizer.log',level=logging.INFO)		
	
	for IMG in IMG_ARR:
		IMG_NAME = '../img/' + IMG + '.jpg' 

		r = Resizer(IMG_NAME)
		logging.info('Reading file: %s\t%s', IMG_NAME, r.raw.shape)
		#r.show(winName='orig')
		t1 = time.clock()	
		r.removeSeams(k=SEAM_K, seam=SEAM)
		t2 = time.clock()
		r.show(winName=IMG+'-Rem-L1-a-slow-'+SEAM+'.jpg', save=True)
		print 'Removed', SEAM_K, SEAM, 'seams-', IMG, ': ', (t2-t1)

		r = Resizer(IMG_NAME)
		r.show(winName='orig')
		t1 = time.clock()	
		r.addSeams(k=SEAM_K, seam=SEAM)
		t2 = time.clock()
		print 'Added  ', SEAM_K, SEAM, 'seams-', IMG, ': ', (t2-t1)
		r.show(winName=IMG+'-Add-L1-a-slow-'+SEAM+'.jpg', save=True)
		logging.info('=========================================')
	cv2.waitKey(0)

