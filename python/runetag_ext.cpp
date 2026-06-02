#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <NTL/tools.h>

#include "runetag.hpp"
#include "ellipsefitter.hpp"

namespace py = pybind11;

void dummyNTLErrorCallback(const char* s) {
    // Do nothing to suppress the spam
}

class FastRuneTagDetector {
public:
    cv::runetag::MarkerDetector* pDetector;
    double cx_val;
    double cy_val;

    FastRuneTagDetector(const std::vector<std::string>& model_paths, double cx = 400.0, double cy = 300.0) {
        NTL::ErrorMsgCallback = dummyNTLErrorCallback;
        cx_val = cx;
        cy_val = cy;
        cv::Mat intrinsics = cv::Mat::eye(3, 3, CV_64FC1);
        intrinsics.at<double>(0, 2) = cx;
        intrinsics.at<double>(1, 2) = cy;
        // Approximate focal length based on typical 800px width camera
        intrinsics.at<double>(0, 0) = 800.0;
        intrinsics.at<double>(1, 1) = 800.0;

        pDetector = new cv::runetag::MarkerDetector(intrinsics);
        for(const auto& model : model_paths) {
            pDetector->addModelsFromFile(model);
        }
    }

    ~FastRuneTagDetector() {
        if(pDetector) delete pDetector;
    }

    py::list detect(py::array_t<uint8_t> input, float minarea, float maxarea, float minroundness, double cx, double cy, double fx, double fy, double offset_x = 0.0, double offset_y = 0.0) {
        py::buffer_info buf = input.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        int channels = buf.ndim == 3 ? buf.shape[2] : 1;
        
        cv::Mat img(rows, cols, channels == 3 ? CV_8UC3 : CV_8UC1, (uint8_t*)buf.ptr);
        
        cv::Mat process_img;
        if(channels == 3) {
            process_img = img;
        } else {
            cv::cvtColor(img, process_img, cv::COLOR_GRAY2RGB);
        }

        // Update intrinsics dynamically to support cropped and full-res frames
        pDetector->intrinsics.at<double>(0, 2) = cx;
        pDetector->intrinsics.at<double>(1, 2) = cy;
        pDetector->intrinsics.at<double>(0, 0) = fx;
        pDetector->intrinsics.at<double>(1, 1) = fy;

        cv::runetag::EllipseDetector ellipseDetector(
            10, 10000, minarea, maxarea, minroundness, 0.3f, -1.5f);

        std::vector<cv::RotatedRect> foundEllipses;
        ellipseDetector.detectEllipses(process_img, foundEllipses);

        std::cout << "[RUNEtag] Crop " << cols << "x" << rows 
                  << " (cx=" << cx << ", cy=" << cy << ", fx=" << fx << ")"
                  << " | MinArea: " << minarea << " MaxArea: " << maxarea
                  << " | Ellipses found: " << foundEllipses.size() << std::endl;

        std::vector<cv::runetag::MarkerDetected> tags_found;
        pDetector->dbgimage = process_img.clone();
        
        try {
            pDetector->detectMarkers(foundEllipses, tags_found);
        } catch (...) {
            return py::list();
        }

        py::list results;
        for(size_t i = 0; i < tags_found.size(); ++i) {
            py::dict res;
            res["idx"] = tags_found[i].associatedModel()->getIDX();
            
            double sum_x = 0;
            double sum_y = 0;
            int count = 0;
            
            for(size_t slot=0; slot<tags_found[i].getNumSlots(); ++slot) {
                if(!tags_found[i].getSlot(slot).discarded() && tags_found[i].getSlot(slot).value()) {
                    cv::Point2d p2d = tags_found[i].getSlot(slot).getPayload()->getCenter() + cv::Point2d(cx, cy);
                    sum_x += p2d.x;
                    sum_y += p2d.y;
                    count++;
                }
            }
            if(count > 0) {
                res["x"] = (sum_x / count) + offset_x;
                res["y"] = (sum_y / count) + offset_y;
            } else {
                res["x"] = cx + offset_x;
                res["y"] = cy + offset_y;
            }
            results.append(res);
        }
        return results;
    }
};

PYBIND11_MODULE(runetag_ext, m) {
    py::class_<FastRuneTagDetector>(m, "FastRuneTagDetector")
        .def(py::init<const std::vector<std::string>&, double, double>(), 
             py::arg("model_paths"), py::arg("cx")=400.0, py::arg("cy")=300.0)
        .def("detect", &FastRuneTagDetector::detect, 
             py::arg("input"), py::arg("minarea")=100.0, py::arg("maxarea")=10000.0, py::arg("minroundness")=0.3, py::arg("cx")=400.0, py::arg("cy")=300.0, py::arg("fx")=800.0, py::arg("fy")=800.0, py::arg("offset_x")=0.0, py::arg("offset_y")=0.0);
}
