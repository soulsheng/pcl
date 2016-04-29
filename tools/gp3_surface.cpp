/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_handlers.h>


typedef pcl::visualization::PointCloudColorHandler<pcl::PCLPointCloud2> ColorHandler;
typedef ColorHandler::Ptr ColorHandlerPtr;
typedef ColorHandler::ConstPtr ColorHandlerConstPtr;

typedef pcl::visualization::PointCloudGeometryHandler<pcl::PCLPointCloud2> GeometryHandler;
typedef GeometryHandler::Ptr GeometryHandlerPtr;
typedef GeometryHandler::ConstPtr GeometryHandlerConstPtr;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

double default_mu = 100.0;
double default_radius = 100.0;

void
printHelp (int, char **argv)
{
  print_error ("Syntax is: %s input.pcd output.vtk <options>\n", argv[0]);
  print_info ("  where options are:\n");
  print_info ("                     -radius X = use a radius of Xm around each point to determine the neighborhood (default: "); 
  print_value ("%f", default_radius); print_info (")\n");
  print_info ("                     -mu X     = set the multipler of the nearest neighbor distance to obtain the final search radius (default: "); 
  print_value ("%f", default_mu); print_info (")\n");
}

bool
loadCloud (const std::string &filename, PointCloud<PointXYZ> &cloud)
{
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (loadPCDFile<PointXYZ> (filename, cloud) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}

void
compute (const PointCloud<PointNormal>::Ptr &input, pcl::PolygonMesh &output,
         double mu, double radius)
{
  // Estimate
  TicToc tt;
  tt.tic ();
  
  print_highlight (stderr, "Computing ");

  PointCloud<PointNormal>::Ptr cloud (new PointCloud<PointNormal> ());
  for (size_t i = 0; i < input->size (); ++i)
    if (pcl_isfinite (input->points[i].x))
      cloud->push_back (input->points[i]);

  cloud->width = static_cast<uint32_t> (cloud->size ());
  cloud->height = 1;
  cloud->is_dense = true;

  GreedyProjectionTriangulation<PointNormal> gpt;
  gpt.setSearchMethod (pcl::search::KdTree<pcl::PointNormal>::Ptr (new pcl::search::KdTree<pcl::PointNormal>));
  gpt.setInputCloud (cloud);
  gpt.setSearchRadius (radius);
  gpt.setMu (mu);

  gpt.reconstruct (output);

  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%lu", output.polygons.size ()); print_info (" polygons]\n");
}

void
saveCloud (const std::string &filename, const pcl::PolygonMesh &output)
{
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());
  saveVTKFile (filename, output);

  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%lu", output.polygons.size ()); print_info (" polygons]\n");
}


void normalEstimation( PointCloud<PointXYZ>::Ptr &cloud, PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals )
{
	// Normal estimation*
	NormalEstimation<PointXYZ, Normal> n;
	PointCloud<Normal>::Ptr normals (new PointCloud<Normal>);
	search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ>);
	tree->setInputCloud (cloud);
	n.setInputCloud (cloud);
	n.setSearchMethod (tree);
	n.setKSearch (20);
	n.compute (*normals);
	//* normals should not contain the point normals + surface curvatures

	// Concatenate the XYZ and normal fields*
	concatenateFields (*cloud, *normals, *cloud_with_normals);
	//* cloud_with_normals = cloud + normals
}

/* ---[ */
int
main (int argc, char** argv)
{
  print_info ("Perform surface triangulation using pcl::GreedyProjectionTriangulation. For more information, use: %s -h\n", argv[0]);
#if 0
  if (argc < 3)
  {
    printHelp (argc, argv);
    return (-1);
  }

  // Parse the command line arguments for .pcd files
  std::vector<int> pcd_file_indices = parse_file_extension_argument (argc, argv, ".pcd");
  if (pcd_file_indices.size () != 1)
  {
    print_error ("Need one input PCD file to continue.\n");
    return (-1);
  }
  std::vector<int> vtk_file_indices = parse_file_extension_argument (argc, argv, ".vtk");
  if (vtk_file_indices.size () != 1)
  {
    print_error ("Need one output VTK file to continue.\n");
    return (-1);
  }
#endif
  // Command line parsing
  double mu = default_mu;
  double radius = default_radius;
  parse_argument (argc, argv, "-mu", mu);
  parse_argument (argc, argv, "-radius", radius);

  // Load the first file
  Eigen::Vector4f translation;
  Eigen::Quaternionf rotation;
  pcl::PCLPointCloud2::Ptr cloud_color (new pcl::PCLPointCloud2);
  if (loadPCDFile ("2.0-f.pcd", *cloud_color, translation, rotation) < 0)
	  return (-1);

  PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);
  fromPCLPointCloud2(*cloud_color, *cloud);

  PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new PointCloud<PointNormal>);
  normalEstimation(cloud, cloud_with_normals);

  // Perform the surface triangulation
  pcl::PolygonMesh output;
  compute (cloud_with_normals, output, mu, radius);

  // Save into the second file
  //saveCloud (argv[vtk_file_indices[0]], output);

  visualization::PCLVisualizer viewer ("Triangular Mesh");
  viewer.addPolygonMesh(output,"Triangular Mesh");

  ColorHandlerPtr color_handler;
  GeometryHandlerPtr geometry_handler;
  color_handler.reset (new pcl::visualization::PointCloudColorHandlerRGBField<pcl::PCLPointCloud2> ( cloud_color ) );
  geometry_handler.reset (new pcl::visualization::PointCloudGeometryHandlerXYZ<pcl::PCLPointCloud2> ( cloud_color ) );
  // Add the cloud to the renderer
  viewer.addPointCloud (cloud_color, geometry_handler, color_handler, translation, rotation, "cloud" );
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
  viewer.spin ();

  system("pause");
}

