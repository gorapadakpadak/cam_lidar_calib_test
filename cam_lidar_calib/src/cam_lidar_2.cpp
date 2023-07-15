#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Geometry>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <iostream>
#include <fstream>

#include <vision_msgs/Detection2DArray.h> //추가

// PointCloud2 and Image message synchronization
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image>
    SyncPolicy;


class LidarImageProjection {
private:
    ros::NodeHandle nh;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Synchronizer<SyncPolicy> *sync;
    ros::Publisher cloud_pub;
    ros::Publisher image_pub;
    ros::Subscriber detection_sub;
    cv::Mat c_R_l, tvec;
    cv::Mat rvec;
    std::string result_str;
    Eigen::Matrix4d C_T_L, L_T_C;
    Eigen::Matrix3d C_R_L, L_R_C;
    Eigen::Quaterniond C_R_L_quatn, L_R_C_quatn;
    Eigen::Vector3d C_t_L, L_t_C;
    bool project_only_plane;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;
    std::vector<cv::Point3d> objectPoints_L, objectPoints_C;
    std::vector<cv::Point2d> imagePoints;
    sensor_msgs::PointCloud2 out_cloud_ros;
    std::string lidar_frameId;
    std::string camera_in_topic;
    std::string lidar_in_topic;
    pcl::PointCloud<pcl::PointXYZRGB> out_cloud_pcl;
    std::vector<vision_msgs::Detection2D> bounding_boxes;
    cv::Mat image_in;
    int dist_cut_off;
    std::string cam_config_file_path;
    int image_width, image_height;
    std::string camera_name;

public:
    lidarImageProjection() {
        camera_in_topic = readParam<std::string>(nh, "camera_in_topic");
        lidar_in_topic = readParam<std::string>(nh, "lidar_in_topic");
        dist_cut_off = readParam<int>(nh, "dist_cut_off");
        camera_name = readParam<std::string>(nh, "camera_name");
        detection_sub = nh.subscribe("/yolov7_detections", 1, &LidarImageProjection::detectionCallback, this); //바운딩박스 구독
        cloud_sub =  new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, lidar_in_topic, 1);
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, camera_in_topic, 1);
        std::string lidarOutTopic = camera_in_topic + "/velodyne_out_cloud";
        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(lidarOutTopic, 1);
        std::string imageOutTopic = camera_in_topic + "/projected_image";
        image_pub = nh.advertise<sensor_msgs::Image>(imageOutTopic, 1);

        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *cloud_sub, *image_sub);
        sync->registerCallback(boost::bind(&lidarImageProjection::callback, this, _1, _2));

        C_T_L = Eigen::Matrix4d::Identity();
        c_R_l = cv::Mat::zeros(3, 3, CV_64F);
        tvec = cv::Mat::zeros(3, 1, CV_64F);

        result_str = readParam<std::string>(nh, "result_file");
        project_only_plane = readParam<bool>(nh, "project_only_plane");

        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(5, 1, CV_64F);

        std::ifstream myReadFile(result_str.c_str());
        std::string word;
        int i = 0;
        int j = 0;
        while (myReadFile >> word){
            C_T_L(i, j) = atof(word.c_str());
            j++;
            if(j>3) {
                j = 0;
                i++;
            }
        }
        L_T_C = C_T_L.inverse();

        C_R_L = C_T_L.block(0, 0, 3, 3);
        C_t_L = C_T_L.block(0, 3, 3, 1);

        L_R_C = L_T_C.block(0, 0, 3, 3);
        L_t_C = L_T_C.block(0, 3, 3, 1);

        cv::eigen2cv(C_R_L, c_R_l);
        C_R_L_quatn = Eigen::Quaterniond(C_R_L);
        L_R_C_quatn = Eigen::Quaterniond(L_R_C);
        cv::Rodrigues(c_R_l, rvec);
        cv::eigen2cv(C_t_L, tvec);

        cam_config_file_path = readParam<std::string>(nh, "cam_config_file_path");
        readCameraParams(cam_config_file_path,
                         image_height,
                         image_width,
                         distCoeff,
                         projection_matrix);
    }

    void readCameraParams(std::string cam_config_file_path,
                      int &image_height,
                      int &image_width,
                      cv::Mat &D,
                      cv::Mat &K) {
        cv::FileStorage fs_cam_config(cam_config_file_path, cv::FileStorage::READ);
        if (!fs_cam_config.isOpened())
            std::cerr << "Error: Wrong path: " << cam_config_file_path << std::endl;
        fs_cam_config["image_height"] >> image_height;
        fs_cam_config["image_width"] >> image_width;
        fs_cam_config["k1"] >> D.at<double>(0);
        fs_cam_config["k2"] >> D.at<double>(1);
        fs_cam_config["p1"] >> D.at<double>(2);
        fs_cam_config["p2"] >> D.at<double>(3);
        fs_cam_config["k3"] >> D.at<double>(4);
        fs_cam_config["fx"] >> K.at<double>(0, 0);
        fs_cam_config["fy"] >> K.at<double>(1, 1);
        fs_cam_config["cx"] >> K.at<double>(0, 2);
        fs_cam_config["cy"] >> K.at<double>(1, 2);
    }

    template <typename T>
    T readParam(ros::NodeHandle &n, std::string name) {
        T ans;
        if (n.getParam(name, ans)) {
            ROS_INFO_STREAM("Loaded " << name << ": " << ans);
        } else {
            ROS_ERROR_STREAM("Failed to load " << name);
            n.shutdown();
        }
        return ans;
    }

    void detectionCallback(const vision_msgs::Detection2DArray::ConstPtr& msg) {
        bounding_boxes = msg->detections;
    }


    pcl::PointCloud<pcl::PointXYZ>::Ptr planeFilter(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *in_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PassThrough<pcl::PointXYZ> pass_x;
    pass_x.setInputCloud(in_cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(0.0, 5.0);
    pass_x.filter(*cloud_filtered_x);
    pcl::PassThrough<pcl::PointXYZ> pass_y;
    pass_y.setInputCloud(cloud_filtered_x);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(-1.25, 1.25);
    pass_y.filter(*cloud_filtered_y);

    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
        new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_y));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
    ransac.setDistanceThreshold(0.01);
    ransac.computeModel();
    std::vector<int> inliers_indices;
    ransac.getInliers(inliers_indices);
    pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered_y, inliers_indices, *plane);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(plane);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1);
    sor.filter(*plane_filtered);

    // Filter the points based on bounding boxes
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const pcl::PointXYZ& point : plane_filtered->points) {
        if (isPointInsideBoundingBoxes(point)) {
            filtered_cloud->points.push_back(point);
        }
    }
    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = true;

    return filtered_cloud;
}


    bool isPointInsideBoundingBoxes(const pcl::PointXYZ& point) {
        // Iterate over the bounding boxes
        for (const vision_msgs::Detection2D& box : bounding_boxes) {
            // Get the bounding box coordinates
            float x_min = box.bbox.center.x - box.bbox.size_x / 2.0;
            float x_max = box.bbox.center.x + box.bbox.size_x / 2.0;
            float y_min = box.bbox.center.y - box.bbox.size_y / 2.0;
            float y_max = box.bbox.center.y + box.bbox.size_y / 2.0;

            // Check if the point falls within the bounding box
            if (point.x >= x_min && point.x <= x_max && point.y >= y_min && point.y <= y_max) {
                return true;
            }
        }

        return false;
    }



    //입력 RGB 이미지(cv::Mat rgb)와 2D 포인트(cv::Point2d xy_f)를 파라미터로 받아 cv::Vec3b 색상 값을 반환
    cv::Vec3b atf(cv::Mat rgb, cv::Point2d xy_f){
        //각 채널(R, G, B)의 색상 값 합계를 저장
        cv::Vec3i color_i;
        // 채널 0으로 초기화
        color_i.val[0] = color_i.val[1] = color_i.val[2] = 0;

        //xy_f에서 x 및 y 값을 추출
        int x = xy_f.x;
        int y = xy_f.y;

        // 2x2  중첩
        //현재 (x+col, y+row) 좌표가 이미지 경계 내에 있는지 확인
        for (int row = 0; row <= 1; row++){
            for (int col = 0; col <= 1; col++){
                if((x+col)< rgb.cols && (y+row) < rgb.rows) {
                    //좌표가 유효하면 함수는 입력 이미지에서 현재 위치의 픽셀 색상(cv::Vec3b c)을 검색
                    cv::Vec3b c = rgb.at<cv::Vec3b>(cv::Point(x + col, y + row));
                    for (int i = 0; i < 3; i++){
                        //각 채널의 색상 값(c.val[i])을 color_i.val[i]에 누적
                        color_i.val[i] += c.val[i];
                    }
                }
            }
        }

        //평균 색상 값을 저장
        cv::Vec3b color;
        for (int i = 0; i < 3; i++){
            //채널 합계를 4(2x2)로 나누고 평균값을 해당 채널에 색으로 할당
            color.val[i] = color_i.val[i] / 4;
        }
        return color;
    }

    //라이다 프레임과 카메라 프레임 사이의 변환
    void publishTransforms() {
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        //라이다-카메라 간 회전 쿼터니언 변환
        tf::quaternionEigenToTF(L_R_C_quatn, q);
        //라이다 변환 벡터L_T_C
        transform.setOrigin(tf::Vector3(L_t_C(0), L_t_C(1), L_t_C(2)));
        transform.setRotation(q);
        //회전과 이동
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), lidar_frameId, camera_name));
    }
    

    //atf 함수에서 얻은 RGB 값을 기반으로 컬러 포인트로 out_cloud_pcl 포인트 클라우드 구성
    void colorPointCloud() {
        //out_cloud_pcl 포인트 클라우드의 기존 포인트를 지우고 objectPoints_L의 수에 맞게 크기를 조정
        out_cloud_pcl.points.clear();
        out_cloud_pcl.resize(objectPoints_L.size());
        
        // atf 함수에서 RGB 색상 값을 검색
        for(size_t i = 0; i < objectPoints_L.size(); i++) {
            cv::Vec3b rgb = atf(image_in, imagePoints[i]);
            //구한 RGB 값을 사용하여 새로운 pcl::PointXYZRGB 포인트(pt_rgb)를 생성
            pcl::PointXYZRGB pt_rgb(rgb.val[2], rgb.val[1], rgb.val[0]);
            //pt_rgb 포인트의 XYZ 좌표는 objectPoints_L의 값으로 설정
            pt_rgb.x = objectPoints_L[i].x;
            pt_rgb.y = objectPoints_L[i].y;
            pt_rgb.z = objectPoints_L[i].z;
            //pt_rgb 포인트가 out_cloud_pcl 포인트 클라우드에 추가
            out_cloud_pcl.push_back(pt_rgb);
        }
    }

    void colorLidarPointsOnImage(double min_range,
            double max_range) {
        //각 imagePoints(2D 이미지 포인트)를 반복하고 objectPoints_C에서 해당 XYZ 좌표를 검색
        for(size_t i = 0; i < imagePoints.size(); i++) {
            double X = objectPoints_C[i].x;
            double Y = objectPoints_C[i].y;
            double Z = objectPoints_C[i].z;
            double range = sqrt(X*X + Y*Y + Z*Z);
            //지정된 범위(최소 범위 및 최대 범위) 내에서 선형 보간을 사용하여 원의 빨간색 및 녹색 색상 값을 계산
            //가까운 지점일수록 빨간색 커짐
            double red_field = 255*(range - min_range)/(max_range - min_range);
            //먼 지점일수록 초록색 값 커짐
            double green_field = 255*(max_range - range)/(max_range - min_range);
            //image_in 이미지에 반지름이 2인 원을 그림
            //가까우면 빨간색, 멀면 초록색
            cv::circle(image_in, imagePoints[i], 2,
                       CV_RGB(red_field, green_field, 0), -1, 1, 0);
        }
    }
    
    void LidarImageProjection::callback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg, const sensor_msgs::Image::ConstPtr& image_msg) {
    lidar_frameId = cloud_msg->header.frame_id;
    objectPoints_L.clear();
    objectPoints_C.clear();
    imagePoints.clear();
    publishTransforms();
    image_in = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    double fov_x, fov_y;
    fov_x = 2*atan2(image_width, 2*projection_matrix.at<double>(0, 0))*180/CV_PI;
    fov_y = 2*atan2(image_height, 2*projection_matrix.at<double>(1, 1))*180/CV_PI;
    double max_range, min_range;
    max_range = -INFINITY;
    min_range = INFINITY;

    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (project_only_plane) {
        in_cloud = planeFilter(cloud_msg);
        for (size_t i = 0; i < in_cloud->points.size(); i++) {
            objectPoints_L.push_back(cv::Point3d(in_cloud->points[i].x, in_cloud->points[i].y, in_cloud->points[i].z));
        }
        cv::projectPoints(objectPoints_L, rvec, tvec, projection_matrix, distCoeff, imagePoints, cv::noArray());
    } else {
        pcl::PCLPointCloud2 *cloud_in = new pcl::PCLPointCloud2;
        pcl_conversions::toPCL(*cloud_msg, *cloud_in);
        pcl::fromPCLPointCloud2(*cloud_in, *in_cloud);

        for (size_t i = 0; i < in_cloud->points.size(); i++) {
            if (in_cloud->points[i].x < 0 || in_cloud->points[i].x > dist_cut_off)
                continue;

            Eigen::Vector4d pointCloud_L;
            pointCloud_L[0] = in_cloud->points[i].x;
            pointCloud_L[1] = in_cloud->points[i].y;
            pointCloud_L[2] = in_cloud->points[i].z;
            pointCloud_L[3] = 1;

            Eigen::Vector3d pointCloud_C;
            pointCloud_C = C_T_L.block(0, 0, 3, 4)*pointCloud_L;

            double X = pointCloud_C[0];
            double Y = pointCloud_C[1];
            double Z = pointCloud_C[2];

            double Xangle = atan2(X, Z)*180/CV_PI;
            double Yangle = atan2(Y, Z)*180/CV_PI;

            if (Xangle < -fov_x/2 || Xangle > fov_x/2)
                continue;

            if (Yangle < -fov_y/2 || Yangle > fov_y/2)
                continue;

            double range = sqrt(X*X + Y*Y + Z*Z);

            if (range > max_range) {
                max_range = range;
            }
            if (range < min_range) {
                min_range = range;
            }

            objectPoints_L.push_back(cv::Point3d(pointCloud_L[0], pointCloud_L[1], pointCloud_L[2]));
            objectPoints_C.push_back(cv::Point3d(X, Y, Z));
        }
        cv::projectPoints(objectPoints_L, rvec, tvec, projection_matrix, distCoeff, imagePoints, cv::noArray());
    }

    // Filter lidar points based on bounding boxes
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& bbox : bounding_boxes) {
        float x1 = bbox.bbox.center.x - bbox.bbox.size_x / 2;
        float y1 = bbox.bbox.center.y - bbox.bbox.size_y / 2;
        float x2 = bbox.bbox.center.x + bbox.bbox.size_x / 2;
        float y2 = bbox.bbox.center.y + bbox.bbox.size_y / 2;

        for (size_t i = 0; i < in_cloud->points.size(); i++) {
            float x = in_cloud->points[i].x;
            float y = in_cloud->points[i].y;

            if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                filtered_cloud->points.push_back(in_cloud->points[i]);
            }
        }
    }

    objectPoints_L.clear();
    for (size_t i = 0; i < filtered_cloud->points.size(); i++) {
        objectPoints_L.push_back(cv::Point3d(filtered_cloud->points[i].x, filtered_cloud->points[i].y, filtered_cloud->points[i].z));
    }

    cv::projectPoints(objectPoints_L, rvec, tvec, projection_matrix, distCoeff, imagePoints, cv::noArray());

    colorPointCloud();
    pcl::toROSMsg(out_cloud_pcl, out_cloud_ros);
    out_cloud_ros.header.frame_id = cloud_msg->header.frame_id;
    out_cloud_ros.header.stamp = cloud_msg->header.stamp;
    cloud_pub.publish(out_cloud_ros);

    colorLidarPointsOnImage(min_range, max_range);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_in).toImageMsg();
    image_pub.publish(msg);
}
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_image_projection");
    LidarImageProjection lidar_image_projection;
    ros::spin();
    return 0;
}
