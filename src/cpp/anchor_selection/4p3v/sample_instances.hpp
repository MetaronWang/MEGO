// Problem-solution sampler for the 4 point problem (Sec. 3)
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int id;
    double x;
    double y;
    double z;
} point3D;

typedef struct {
    int id;
    double w;
    double h;
    double fx;
    double fy;
    double px;
    double py;
} camera;

typedef struct {
    double x;
    double y;
    int pt_id;
} point2D;

typedef struct {
    int id;
    double qw;
    double qx;
    double qy;
    double qz;
    double tx;
    double ty;
    double tz;
    int cam_id;
} image;

//loads a model in the COLMAP .txt format which is stored in data_folder
void load_model(std::string data_folder,
                std::vector<camera> &cams,
                std::unordered_map<int, int> &id2cm,
                std::vector<image> &imgs,
                std::unordered_map<int, int> &id2img,
                std::vector<std::vector<point2D>> &obs,
                std::vector<point3D> &pts,
                std::unordered_map<int, int> &id2pt) {
    std::string line;

    //load the cameras
    std::ifstream fc;
    fc.open(data_folder + "/cameras.txt");
    std::getline(fc, line);
    std::getline(fc, line);
    fc >> line;
    fc >> line;
    fc >> line;
    fc >> line;
    int nc;
    fc >> nc;
    std::getline(fc, line);

    std::cerr << nc << " camera models\n";
    for (int i = 0; i < nc; ++i) {
        camera cam;

        int id;
        fc >> id;
        cam.id = id;
        id2cm[id] = i;

        fc >> line;

        double x;
        fc >> x;
        cam.w = x;

        fc >> x;
        cam.h = x;

        fc >> x;
        cam.fx = x;

        fc >> x;
        cam.fy = x;

        fc >> x;
        cam.px = x;

        fc >> x;
        cam.py = x;

        cams.push_back(cam);
    }

    fc.close();

    //load the views
    std::ifstream fi;
    fi.open(data_folder + "/images.txt");

    std::getline(fi, line);
    std::getline(fi, line);
    std::getline(fi, line);
    fi >> line;
    fi >> line;
    fi >> line;
    fi >> line;
    int ni;
    fi >> ni;
    std::getline(fi, line);

    std::cerr << ni << " views\n";
    for (int i = 0; i < ni; ++i) {
        image img;

        int id;
        fi >> id;
        img.id = id;
        id2img[id] = i;

        double x;
        fi >> x;
        img.qw = x;

        fi >> x;
        img.qx = x;

        fi >> x;
        img.qy = x;

        fi >> x;
        img.qz = x;

        fi >> x;
        img.tx = x;

        fi >> x;
        img.ty = x;

        fi >> x;
        img.tz = x;

        fc >> id;
        img.cam_id = id;

        std::getline(fi, line);

        imgs.push_back(img);

        //load the observations
        std::getline(fi, line);
        std::stringstream ss;
        ss << line;
        std::vector<point2D> ob;
        while (ss >> x) {
            double y;
            ss >> y;
            ss >> id;
            if (id >= 0) {

                point2D p2;
                p2.x = x;
                p2.y = y;
                p2.pt_id = id;

                ob.push_back(p2);
            }
        }
        obs.push_back(ob);

    }

    fi.close();

    //load the 3D points
    std::ifstream fp;
    fp.open(data_folder + "/points3D.txt");

    std::getline(fp, line);
    std::getline(fp, line);
    fp >> line;
    fp >> line;
    fp >> line;
    fp >> line;
    int np;
    fp >> np;
    std::getline(fp, line);

    std::cerr << np << " 3D points\n";
    for (int i = 0; i < np; ++i) {
        point3D pt;

        int id;
        fp >> id;
        pt.id = id;
        id2pt[id] = i;

        double x;
        fp >> x;
        pt.x = x;

        fp >> x;
        pt.y = x;

        fp >> x;
        pt.z = x;

        pts.push_back(pt);

        std::getline(fp, line);
    }

    fp.close();
}

Eigen::Matrix3d q2R(double qr, double qi, double qj, double qk) {
    Eigen::Matrix3d R;
    double n = qr * qr + qi * qi + qj * qj + qk * qk;
    double s = 1 / n;
    R(0, 0) = 1 - 2 * s * (qj * qj + qk * qk);
    R(0, 1) = 2 * s * (qi * qj - qk * qr);
    R(0, 2) = 2 * s * (qi * qk + qj * qr);

    R(1, 0) = 2 * s * (qi * qj + qk * qr);
    R(1, 1) = 1 - 2 * s * (qi * qi + qk * qk);
    R(1, 2) = 2 * s * (qj * qk - qi * qr);

    R(2, 0) = 2 * s * (qi * qk - qj * qr);
    R(2, 1) = 2 * s * (qj * qk + qi * qr);
    R(2, 2) = 1 - 2 * s * (qi * qi + qj * qj);

    return R;
}

void order_points(std::vector<Eigen::Vector3d> &P, int *perm4, int ix) {
    double angle1 = std::atan2(P[ix](1), P[ix](0));
    if (angle1 < 0)
        angle1 = angle1 + 2 * acos(-1.0);

    //obtain the relative angles
    double angles[4];
    for (int i = 0; i < 4; ++i) {
        if (i == ix) {
            angles[ix] = 7;
            continue;
        }
        //if negative, add 2*pi to obtain a positive number
        double cur_ang = std::atan2(P[i](1), P[i](0));
        if (cur_ang < 0)
            cur_ang = cur_ang + 2 * acos(-1.0);
        //subtract the angle of the longest point from other angles (if negative, add 2*pi)
        double ang = cur_ang - angle1;
        if (ang < 0)
            ang = ang + 2 * acos(-1.0);
        angles[i] = ang;
    }

    perm4[3] = ix;

    for (int i = 0; i < 3; ++i) {
        double min = 7;
        int next = -1;
        for (int j = 0; j < 4; ++j) {
            if (angles[j] < min) {
                next = j;
                min = angles[j];
            }
        }
        perm4[i] = next;
        angles[next] = 7;
    }
}

void normalize(std::vector<Eigen::Vector3d> &P, std::vector<Eigen::Vector3d> &Q, std::vector<Eigen::Vector3d> &R,
               std::vector<Eigen::Vector2d> &P1, std::vector<Eigen::Vector2d> &Q1, std::vector<Eigen::Vector2d> &R1,
               Eigen::Matrix3d &CP1, Eigen::Matrix3d &CQ1, Eigen::Matrix3d &CR1, int *perm, int *perm4) {
    //project the points to a sphere and obtain the centroids
    Eigen::Vector3d centroidP = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroidQ = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroidR = Eigen::Vector3d::Zero();
    for (int i = 0; i < 4; ++i) {
        P[i] = P[i] / P[i].norm();
        Q[i] = Q[i] / Q[i].norm();
        R[i] = R[i] / R[i].norm();
        centroidP = centroidP + P[i];
        centroidQ = centroidQ + Q[i];
        centroidR = centroidR + R[i];
    }
    centroidP = 0.25 * centroidP;
    centroidQ = 0.25 * centroidQ;
    centroidR = 0.25 * centroidR;
    centroidP = centroidP / centroidP.norm();
    centroidQ = centroidQ / centroidQ.norm();
    centroidR = centroidR / centroidR.norm();

    //identify the first point and view
    int ix;
    int view;
    int second;
    int first;
    double best = 99999999;
    for (int i = 0; i < 4; ++i) {
        double ang = P[i].transpose() * centroidP;
        if (ang < best) {
            best = ang;
            ix = i;
            view = 0;
        }

        ang = Q[i].transpose() * centroidQ;
        if (ang < best) {
            best = ang;
            ix = i;
            view = 1;
        }

        ang = R[i].transpose() * centroidR;
        if (ang < best) {
            best = ang;
            ix = i;
            view = 2;
        }
    }

    //order the other two views
    if (!view) {
        double ang1 = Q[ix].transpose() * centroidQ;
        double ang2 = R[ix].transpose() * centroidR;
        if (ang1 < ang2) {
            second = 1;
            first = 2;
        } else {
            second = 2;
            first = 1;
        }
    } else if (view == 1) {
        double ang0 = P[ix].transpose() * centroidP;
        double ang2 = R[ix].transpose() * centroidR;
        if (ang0 < ang2) {
            second = 0;
            first = 2;
        } else {
            second = 2;
            first = 0;
        }
    } else {
        double ang0 = P[ix].transpose() * centroidP;
        double ang1 = Q[ix].transpose() * centroidQ;
        if (ang0 < ang1) {
            second = 0;
            first = 1;
        } else {
            second = 1;
            first = 0;
        }
    }

    //rotate the centroid to zero and the given point to y axis
    Eigen::Vector3d p0 = centroidP / centroidP.norm();
    Eigen::Vector3d p1 = p0.cross(P[ix]);
    p1 = p1 / p1.norm();
    Eigen::Vector3d p2 = p0.cross(p1);
    p2 = p2 / p2.norm();
    Eigen::Matrix3d Zp;
    Zp << p0, p1, p2;

    Eigen::Vector3d q0 = centroidQ / centroidQ.norm();
    Eigen::Vector3d q1 = q0.cross(Q[ix]);
    q1 = q1 / q1.norm();
    Eigen::Vector3d q2 = q0.cross(q1);
    q2 = q2 / q2.norm();
    Eigen::Matrix3d Zq;
    Zq << q0, q1, q2;

    Eigen::Vector3d r0 = centroidR / centroidR.norm();
    Eigen::Vector3d r1 = r0.cross(R[ix]);
    r1 = r1 / r1.norm();
    Eigen::Vector3d r2 = r0.cross(r1);
    r2 = r2 / r2.norm();
    Eigen::Matrix3d Zr;
    Zr << r0, r1, r2;

    Eigen::Matrix3d ZZ;
    ZZ << 0, 0, -1, 0, 1, 0, 1, 0, 0;
    Eigen::Matrix3d CP = ZZ * Zp.transpose();
    Eigen::Matrix3d CQ = ZZ * Zq.transpose();
    Eigen::Matrix3d CR = ZZ * Zr.transpose();

    //rotate the points and project them back to the plane
    for (int i = 0; i < 4; ++i) {
        P[i] = CP * P[i];
        P[i] = P[i] / P[i](2);

        Q[i] = CQ * Q[i];
        Q[i] = Q[i] / Q[i](2);

        R[i] = CR * R[i];
        R[i] = R[i] / R[i](2);
    }

    perm[first] = 0;
    perm[second] = 1;
    perm[view] = 2;

    //order the points
    if (view == 0)
        order_points(P, perm4, ix);
    else if (view == 1)
        order_points(Q, perm4, ix);
    else
        order_points(R, perm4, ix);

    //permute the view with the longest point
    if (view == 0) //perm[2]==0 means that old view 0 is new view 2, perm4[j]=q means that old pnt q is new pnt j
    {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = P[perm4[j]](0);
            p(1) = P[perm4[j]](1);
            R1[j] = p;
            CR1 = CP;
        }
    } else if (view == 1) {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = Q[perm4[j]](0);
            p(1) = Q[perm4[j]](1);
            R1[j] = p;
            CR1 = CQ;
        }
    } else {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = R[perm4[j]](0);
            p(1) = R[perm4[j]](1);
            R1[j] = p;
            CR1 = CR;
        }
    }

    //permute the view with the second longest point
    if (second == 0) {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = P[perm4[j]](0);
            p(1) = P[perm4[j]](1);
            Q1[j] = p;
            CQ1 = CP;
        }
    } else if (second == 1) {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = Q[perm4[j]](0);
            p(1) = Q[perm4[j]](1);
            Q1[j] = p;
            CQ1 = CQ;
        }
    } else {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = R[perm4[j]](0);
            p(1) = R[perm4[j]](1);
            Q1[j] = p;
            CQ1 = CR;
        }
    }

    //permute the view with the shortest point
    if (first == 0) {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = P[perm4[j]](0);
            p(1) = P[perm4[j]](1);
            P1[j] = p;
            CP1 = CP;
        }
    } else if (first == 1) {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = Q[perm4[j]](0);
            p(1) = Q[perm4[j]](1);
            P1[j] = p;
            CP1 = CQ;
        }
    } else {
        for (int j = 0; j < 4; j++) {
            Eigen::Vector2d p;
            p(0) = R[perm4[j]](0);
            p(1) = R[perm4[j]](1);
            P1[j] = p;
            CP1 = CR;
        }
    }
}

std::vector<std::vector<double>> sample_instances(std::string data_folder, int samples) {
    //run as: 4p3v data_folder num_samples
    //where: data_folder is the folder where the data is located, num_samples is the number of samples per camera pair
    std::vector<std::vector<double>> instances;
    std::cerr << "Extracting data from folder " << data_folder << ".\n";

    std::cerr << samples << " samples\n";

    //init random generator
    srand(time(NULL));

    //load the COLMAP model from the given folder
    std::vector<camera> cams;
    std::unordered_map<int, int> id2cm;
    std::vector<image> imgs;
    std::unordered_map<int, int> id2img;
    std::vector<std::vector<point2D>> obs;
    std::vector<point3D> pts;
    std::unordered_map<int, int> id2pt;
    load_model(data_folder, cams, id2cm, imgs, id2img, obs, pts, id2pt);

    //convert the 3D points to Eigen
    std::vector<Eigen::Vector3d> X(pts.size());
    std::unordered_map<int, int> pnts_map;
    for (unsigned int i = 0; i < pts.size(); i++) {
        Eigen::Vector3d cXh;
        cXh(0) = pts[i].x;
        cXh(1) = pts[i].y;
        cXh(2) = pts[i].z;
        X[i] = cXh;
        pnts_map[pts[i].id] = i;
    }

    //convert the GT absolute poses and reproject the observed points to the cameras
    std::vector<Eigen::Matrix3d> Rs(imgs.size());
    std::vector<Eigen::Vector3d> ts(imgs.size());
    std::vector<std::vector<Eigen::Vector2d>> Us;
    std::vector<double> fs(imgs.size());
    for (int i = 0; i < imgs.size(); ++i) {
        //rotation matrix from quaternion, translation vectors from its coordinates
        Eigen::Matrix3d R = q2R(imgs[i].qw, imgs[i].qx, imgs[i].qy, imgs[i].qz);
        Eigen::Vector3d t(imgs[i].tx, imgs[i].ty, imgs[i].tz);
        Rs[i] = R;
        ts[i] = t;

        //compose K
        Eigen::Matrix3d K;
        K << cams[id2cm[imgs[i].cam_id]].fx, 0, cams[id2cm[imgs[i].cam_id]].px,
                0, cams[id2cm[imgs[i].cam_id]].fy, cams[id2cm[imgs[i].cam_id]].py,
                0, 0, 1;
        Eigen::Matrix3d Km1 = K.inverse();
        fs[i] = cams[id2cm[imgs[i].cam_id]].fx;

        //init the projections to NaN
        std::vector<Eigen::Vector2d> proj(X.size());
        for (unsigned int j = 0; j < X.size(); j++) {
            Eigen::Vector2d c_proj_;
            c_proj_(0) = -999999;// std::numeric_limits<double>::quiet_NaN();
            c_proj_(1) = -999999;//std::numeric_limits<double>::quiet_NaN();
            proj[j] = c_proj_;
        }

        //if exact, project the points, if real, calibrate the projections

        //reproject the observed 3D points to the camera
        for (int j = 0; j < obs[i].size(); ++j) {
            Eigen::Vector3d ptX = X[id2pt[obs[i][j].pt_id]];
            Eigen::Vector3d cal = R * ptX + t;
            proj[id2pt[obs[i][j].pt_id]] = cal.hnormalized();
        }
        Us.push_back(proj);
    }

    //initialize the variables for normalization and for storing the points
    std::vector<Eigen::Vector2d> points1(4);
    std::vector<Eigen::Vector2d> points2(4);
    std::vector<Eigen::Vector2d> points3(4);
    std::vector<Eigen::Vector2d> points1_n(4);
    std::vector<Eigen::Vector2d> points2_n(4);
    std::vector<Eigen::Vector2d> points3_n(4);
    std::vector<double> depths1_n(4);
    std::vector<double> depths2_n(4);
    std::vector<double> depths3_n(4);
    int perm3[3];
    int perm4[4];

    //for every pair of cameras sample the p-s pairs
    for (int i = 0; i < imgs.size(); ++i) {
//        std::cerr << i << "\n";

        std::vector<Eigen::Vector2d> U1 = Us[i];
        for (int j = i + 1; j < imgs.size(); ++j) {
            std::vector<Eigen::Vector2d> U2 = Us[j];
            for (int k = j + 1; k < imgs.size(); ++k) {
                std::vector<Eigen::Vector2d> U3 = Us[k];

                //obtain the points observed by all three cameras
                std::vector<int> ok;
                for (unsigned int l = 0; l < U1.size(); l++) {
                    //if(!std::isnan(U1[l](0)) && !std::isnan(U2[l](0)) && !std::isnan(U3[l](0)))
                    if (U1[l](0) > -9999 && U2[l](0) > -9999 && U3[l](0) > -9999)
                        ok.push_back(l);
                }

                int samples2 = samples;
                if (ok.size() == 5 && samples > 1)
                    samples2 = 1;
                else if (ok.size() == 6 && samples > 6)
                    samples2 = 6;
                else if (ok.size() == 7 && samples > 42)
                    samples2 = 42;
                else if (ok.size() < 10 && samples > 100)
                    samples2 = 100;
                if (!ok.size()) continue;

                //sample the point tuples and obtain the corresponding PS pairs
                for (int a = 0; a < samples2; ++a) {
                    //sample 4 points
                    bool bad_smpl = 0;
                    std::vector<size_t> smpl(5);
                    for (int a = 0; a < 4; ++a) {
                        smpl[a] = rand() % ok.size();
                        for (int b = 0; b < a; ++b) {
                            if (smpl[a] == smpl[b])
                                bad_smpl = 1;
                        }
                    }
                    if (bad_smpl) continue;

                    bool found = 0;
                    for (int b = 0; b < 4; b++) {
                        points1[b] = (U1[ok[smpl[b]]]);
                        points2[b] = (U2[ok[smpl[b]]]);
                        points3[b] = (U3[ok[smpl[b]]]);
                    }

                    //normalize the sampled problem
                    std::vector<Eigen::Vector3d> P(4);
                    std::vector<Eigen::Vector3d> Q(4);
                    std::vector<Eigen::Vector3d> R(4);
                    for (int l = 0; l < 4; ++l) {
                        P[l] = points1[l].homogeneous();
                        Q[l] = points2[l].homogeneous();
                        R[l] = points3[l].homogeneous();
                    }
                    std::vector<Eigen::Vector2d> Pn(4);
                    std::vector<Eigen::Vector2d> Qn(4);
                    std::vector<Eigen::Vector2d> Rn(4);
                    Eigen::Matrix3d CP;
                    Eigen::Matrix3d CQ;
                    Eigen::Matrix3d CR;
                    normalize(P, Q, R, Pn, Qn, Rn, CP, CQ, CR, perm3, perm4);

                    //construct normalized "projection" matrices
                    int ix[3];
                    ix[perm3[0]] = i;
                    ix[perm3[1]] = j;
                    ix[perm3[2]] = k;
                    Eigen::Matrix3d rot1 = CP * Rs[ix[0]];
                    Eigen::Vector3d tran1 = CP * ts[ix[0]];
                    Eigen::Matrix3d rot2 = CQ * Rs[ix[1]];
                    Eigen::Vector3d tran2 = CQ * ts[ix[1]];
                    Eigen::Matrix3d rot3 = CR * Rs[ix[2]];
                    Eigen::Vector3d tran3 = CR * ts[ix[2]];

                    std::vector<double> instance(37);
                    //project the 3D points exactly by the normalized cameras to get the depths and the projections
                    for (int a = 0; a < 4; ++a) {
                        Eigen::Vector3d proj1 = CP * (Rs[ix[0]] * X[ok[smpl[perm4[a]]]] + ts[ix[0]]);
                        Eigen::Vector3d proj2 = CQ * (Rs[ix[1]] * X[ok[smpl[perm4[a]]]] + ts[ix[1]]);
                        Eigen::Vector3d proj3 = CR * (Rs[ix[2]] * X[ok[smpl[perm4[a]]]] + ts[ix[2]]);


                        depths1_n[a] = proj1(2);
                        depths2_n[a] = proj2(2);
                        depths3_n[a] = proj3(2);

                        Eigen::Vector2d p1;
                        p1(0) = proj1(0) / proj1(2);
                        p1(1) = proj1(1) / proj1(2);
                        points1_n[a] = p1;

                        Eigen::Vector2d p2;
                        p2(0) = proj2(0) / proj2(2);
                        p2(1) = proj2(1) / proj2(2);
                        points2_n[a] = p2;

                        Eigen::Vector2d p3;
                        p3(0) = proj3(0) / proj3(2);
                        p3(1) = proj3(1) / proj3(2);
                        points3_n[a] = p3;

                        instance[a] = points1_n[a](0);
                        instance[a + 4] = points1_n[a](1);
                        instance[a + 8] = points2_n[a](0);
                        instance[a + 12] = points2_n[a](1);
                        instance[a + 16] = points3_n[a](0);
                        instance[a + 20] = points3_n[a](1);

                        instance[a + 24] = depths1_n[a];
                        instance[a + 28] = depths2_n[a];
                        instance[a + 32] = depths3_n[a];


                    }
                    instance[36] = 0;
                    instances.push_back(instance);
                }
            }
        }
    }
    return instances;
}

void output_instances(std::vector<std::vector<double>> &instances) {
    std::cout << instances.size() << "\n";
    for (std::vector<double> instance: instances) {
        for (int a = 0; a < 24; ++a) {
            std::cout << std::setprecision(25) << instance[a] << " ";
        }
        for (int a = 0; a < 13; ++a) {
            std::cout << std::setprecision(25) << instance[a + 24] << " ";
        }
        std::cout << "\n";
    }
}
