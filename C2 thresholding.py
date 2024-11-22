# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dd1C3y98w3QS_DnZFggODbuuu8LCPX2B
"""

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
class segmentation:
    def __init__(self):
        s=str(input("Enter the path to image:"))
        self.img1=cv2.imread(s)
        self.img=cv2.cvtColor(self.img1,cv2.COLOR_BGR2GRAY)
    def simple_thresholding(self):
          ret,thre=cv2.threshold(self.img,120,255,cv2.THRESH_BINARY)
          plt.subplot(121)
          plt.imshow(self.img,cmap='gray')
          plt.title("Original image")
          plt.subplot(122)
          plt.imshow(thre,cmap='gray')
          plt.title("Simple thresholding segmentation")
          plt.show()
          dsc=self.DSC(thre,self.img)
          print("DSC VALUE=",dsc)
    def adaptive_thresholding(self):
         thres=cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10)
         plt.subplot(121)
         plt.imshow(self.img,cmap='gray')
         plt.title("Original image")
         plt.subplot(122)
         plt.imshow(thres,cmap='gray')
         plt.title("Adaptive thresholding segmentation")
         plt.show()
         dsc=self.DSC(thres,self.img)
         print("DSC VALUE=",dsc)
    def otsu_thresholding(self):
         (t,threshinv)=cv2.threshold(self.img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
         plt.subplot(121)
         plt.imshow(self.img,cmap='gray')
         plt.title("Original image")
         plt.subplot(122)
         plt.imshow(threshinv,cmap='gray')
         plt.title("Otsu segmentation")
         plt.show()
         dsc=self.DSC(threshinv,self.img)
         print("DSC VALUE=",dsc)
    def DSC(self,img1,img2):
        img1=img1.astype(bool)
        img2=img2.astype(bool)
        inter=np.logical_and(img1,img2)
        dsc=2.*inter.sum() /(img1.sum()+img2.sum())
        return dsc
    def region_growing(self, seed_point):
      x, y = self.img.shape
      print(f"Image dimensions: {x}x{y}")

      segmented = np.zeros((x, y), np.uint8)
      threshold = float(input("Enter the value of threshold: "))

      # Ensure seed_point is within bounds
      if not (0 <= seed_point[0] < x and 0 <= seed_point[1] < y):
          print("Seed point is outside image bounds!")
          return

      seed_value = self.img[seed_point[0], seed_point[1]]
      region = [seed_point]
      visited = np.zeros((x, y), np.bool_)
      visited[seed_point] = True
      segmented[seed_point] = 255

      neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
      while region:
          current = region.pop(0)
          for neighbor in neighbors:
              neigh = (current[0] + neighbor[0], current[1] + neighbor[1])
              if 0 <= neigh[0] < x and 0 <= neigh[1] < y:
                  if not visited[neigh]:
                      if abs(int(self.img[neigh]) - int(seed_value)) < threshold:
                          segmented[neigh] = 255
                          region.append(neigh)
                      visited[neigh] = True

      plt.subplot(121)
      plt.imshow(self.img, cmap='gray')
      plt.title("Original Image")
      plt.subplot(122)
      plt.imshow(segmented, cmap='gray')
      plt.title("Region Growing Segmentation")
      plt.show()

      dsc = self.DSC(segmented, self.img)
      print("DSC VALUE =", dsc)

if __name__=="__main__":
     seg=segmentation()
     while(True):
          ch=int(input('Enter 1 to apply simple thresholding,Enter 2 to apply adaptive thresholding,Enter 3 to apply otsu thresholding,Enter 4 to apply region growing segmentation,Enter 5 to exit:'))
          if ch==1:
               seg.simple_thresholding()
          if ch==2:
               seg.adaptive_thresholding()
          if ch==3:
               seg.otsu_thresholding()
          if ch==4:
               seg.region_growing((20,20))
          if ch==5:
               exit()