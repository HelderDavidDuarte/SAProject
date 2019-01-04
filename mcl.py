#!/usr/bin/env python
from __future__ import division
import rospy
import tf
import tf.transformations as tr
from std_msgs.msg import String, Header, Float32
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point, PoseArray, Pose, Quaternion
from sensor_msgs.msg import LaserScan, Imu, Image
from math import sqrt, cos, sin, pi, log, exp
import numpy as np
PKG = 'mcl_sa'
import roslib; roslib.load_manifest(PKG)
import time

class Particle(object): 
	''' particula definida pela posicao (x,y) e orientacao theta 
	e pelo seu numero de serie id'''
	def __init__(self, x, y, theta, weight):
		self.x = x
		self.y = y
		self.theta = theta
		self.weight = weight

class ParticleFilter(object):

	prob_matrix = None
	
	def __init__(self, occ_grid_map):
		self.num_particles = 500
		self.ogm = occ_grid_map            #msg toda do occ grid
		self.grid_map = np.array(self.ogm.data) 

		#retira a matriz do mapa enviada peloa msg do map_server
		self.grid_map = self.grid_map.reshape((self.ogm.info.height,self.ogm.info.width))
		self.resolution = self.ogm.info.resolution
		self.width = self.ogm.info.width
		self.height = self.ogm.info.height
		# transforma essa matriz 1D numa 2D height x width

		# LIMITES
		self.xmin = self.ogm.info.origin.position.x
		self.xmax = (self.width)*self.resolution + self.xmin
		self.ymin = (self.ogm.info.origin.position.y)
		self.ymax = (self.height)*self.resolution + self.ymin
		self.laser_z_hit = 0.1
		self.laser_z_rand = 0.9
		self.laser_sigma_hit = 0.2
		self.conv=0

		# numero de raios do laser
		self.angle_min = 1
		self.angle_max = 1
		self.angle_increment = 1
		self.ranges= []

		# anterior medida de imu do robot
		self.last_robot_imu = None

		# atual imu do robot
		self.robot_imu = None
		self.particles = []
		self.tempo = 0
		self.tempoanterior = round(time.time() * 1000**2)
		self.dt=0
		self.angularz=0

		rospy.Subscriber("/scan_front", LaserScan, self.laserCallback)
		#rospy.Subscriber('odom',Odometry,self.imuread)
		rospy.Subscriber("/imu/angularz", Float32, self.imuread)
		self.particle_pub = rospy.Publisher("/particles", PoseArray, queue_size = self.num_particles)

		if ParticleFilter.prob_matrix is None:
			self.prob_grid_map()
		#rospy.Subscriber("map",OccupancyGrid, self.mapa_callback)

	def imuread(self, msg):
		self.angularz = msg.data
	#	self.angularz=msg.twist.twist.angular.z

	def laserCallback(self,msg):
		self.angle_min = msg.angle_min
		self.angle_max = msg.angle_max
		self.angle_increment = msg.angle_increment
		self.ranges = msg.ranges

 	def prob_grid_map(self):
 		from sklearn.neighbors import KDTree
 		cell_ocupadas = []
 		cell_todas = []
 		for i in range(self.grid_map.shape[0]):
 			for j in range(self.grid_map.shape[1]):
 				cell_todas.append(self.grid_to_metric(i,j))
 				if self.grid_map[i,j]>90:
 					cell_ocupadas.append(self.grid_to_metric(i,j))
 		kdt = KDTree(cell_ocupadas)
 		dists = kdt.query(cell_todas, k=1)[0][:]
 		probs = np.exp(-(dists**2)/(2*self.laser_sigma_hit**2))
 		probs=probs.reshape(self.ogm.info.height,self.ogm.info.width)
 		ParticleFilter.prob_matrix = probs

	def init_particles(self):
		self.particles=[]
		weight = 1/self.num_particles
		''' Inicia as particulas espalhadas uniformemente com as coord
		da frame do mapa, dentro dos limites e pesos iguais'''
		for i in range(self.num_particles):
			xrand, yrand, theta = self.get_random_state()
			self.particles.append(Particle(xrand, yrand, theta, weight))

	def get_random_state(self):
		while True:
			xrand = np.random.uniform(self.xmin, self.xmax) 
			yrand = np.random.uniform(self.ymin, self.ymax)
			row, col = self.metric_to_grid(xrand, yrand)
			if self.grid_map[row, col] == 0:
				theta = np.random.uniform(-2*pi, 2*pi)
				return xrand, yrand, theta

 	def metric_to_grid(self, x, y):
 		''' converte coordenadas metricas para coord da occ grid'''
 		row = min(max(int((y-self.ymin)/self.resolution), 0), self.height)-1
 		col = min(max(int((x-self.xmin)/self.resolution), 0), self.width)-1
 		return row, col

 	def grid_to_metric(self,row,col):
 		'''Da as coordenadas x e y do centro de uma dada cell'''
 		x = (row+0.5)*self.resolution+self.xmin
 		y = (col+0.5)*self.resolution+self.ymin
 		return x, y

 	#MOTION MODEL
 	def atualizapos(self): 
		for i in range(self.num_particles):
			self.tempo=round(time.time() * 1000**2)
			self.dt=self.tempo-self.tempoanterior
			self.particles[i].x+=np.random.normal(0,0.1)
			self.particles[i].y+=np.random.normal(0,0.1)
			self.particles[i].theta+=self.angularz*self.dt+np.random.normal(0,pi/4)
			self.tempoanterior=self.tempo

 	def scan_to_endpoints(self,x,y,theta):
 		theta_beam = np.arange(self.angle_min, self.angle_max+self.angle_increment, self.angle_increment)
 		ranges = np.array(self.ranges)
 		xs = (x+ranges*np.cos(theta+theta_beam))
 		ys = (y+ranges*np.sin(theta+theta_beam))
 		# clear nan entries
 		xs = xs[np.logical_not(np.isnan(xs))]
 		ys = ys[np.logical_not(np.isnan(ys))]
 		# clear inf entries
 		xs = xs[np.logical_not(np.isinf(xs))]
 		ys = ys[np.logical_not(np.isinf(ys))]
 		return xs, ys

 	def obs_model(self):
 		for i in range(self.num_particles):
 			xs, ys = self.scan_to_endpoints(self.particles[i].x, self.particles[i].y, self.particles[i].theta)
 			total_prob = 0
 			for j in range(0,len(xs),10):
 				row, col = self.metric_to_grid(xs[j],ys[j])
 				probabilidades=ParticleFilter.prob_matrix[row,col]
  				if np.isnan(probabilidades) or self.grid_map[row, col] != 0:
 					probabilidades=0
 				total_prob += np.log(self.laser_z_hit*probabilidades+self.laser_z_rand)
 			self.particles[i].weight *= np.exp(total_prob)

	def publish_particles(self):
		#Converts a particle in the form [x, y, theta] into a Pose object
		Poses = PoseArray()
		Poses.header.frame_id = "/map"
		Poses.header.stamp = rospy.Time.now()
		for i in range(self.num_particles):
			pose = Pose()
			pose.position.x = self.particles[i].x
			pose.position.y = self.particles[i].y
			pose.orientation = Quaternion(*tr.quaternion_from_euler(0,0,self.particles[i].theta))
			Poses.poses.append(pose)
		self.particle_pub.publish(Poses)

	def resample(self):
		new_particles=[]
		weights=[]
		sqweights=0
		weightsum=0
		for k in range(self.num_particles):
			weightsum+=self.particles[k].weight
		#normalizar pesos
		for l in range(self.num_particles):
			sqweights+=(self.particles[l].weight/weightsum)**2
			weights.append(self.particles[l].weight/weightsum)
		# escolher indexes conforme a probabilidade
		if 1/sqweights<self.num_particles*0.5:
			new_index = np.random.choice(self.num_particles,self.num_particles, replace=True, p=weights)
			for i in new_index:
				new_particles.append(Particle(self.particles[i].x+np.random.normal(0,0.1), self.particles[i].y+np.random.normal(0,0.1), self.particles[i].theta, weights[i]))
			self.particles = new_particles
		self.publish_particles()

	def kidnapping(self):
		xvector=[]
 		yvector=[]
 		for i in range(self.num_particles):
 			xvector.append(self.particles[i].x)
 			yvector.append(self.particles[i].y)
		if not ((max(xvector)-min(xvector))<1.5 and (max(yvector)-min(yvector))<1.5):
			self.conv-=1
		elif self.conv<10:
			self.conv+=1
		if self.conv<-10:
			self.conv=0
			self.init_particles()


if __name__ == '__main__' :
	rospy.init_node('mcl')
	# espera pela msg do mapa
	map_msg = rospy.wait_for_message("map", OccupancyGrid)
	pf = ParticleFilter(map_msg)
	# inicializa as particulas em pose random
	pf.init_particles()

	while not rospy.is_shutdown():
		pf.atualizapos()
		pf.obs_model()
		pf.resample()
		pf.kidnapping()
		rospy.sleep(0.3)

	rospy.spin()