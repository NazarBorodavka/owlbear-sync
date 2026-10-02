#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// CCTag includes
#include <cctag/Detection.hpp>
#include <cctag/CCTagMarkersBank.hpp>
#include <boost/ptr_container/ptr_list.hpp>

namespace py = pybind11;
using namespace cctag;

class FastCCTagDetector {
public:
    cctag::Parameters params;
    CCTagMarkersBank bank;

    FastCCTagDetector(int nCrowns = 3) : params(nCrowns), bank(nCrowns) {
    }

    py::list detect(py::array_t<uint8_t> input, float min_ident_proba=1e-6f, double cx=400.0, double cy=300.0, double fx=800.0, double fy=800.0, double offset_x=0.0, double offset_y=0.0) {
        py::buffer_info buf = input.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        int channels = buf.ndim == 3 ? buf.shape[2] : 1;
        
        cv::Mat img(rows, cols, channels == 3 ? CV_8UC3 : CV_8UC1, (uint8_t*)buf.ptr);
        cv::Mat graySrc;
        if(channels == 3) {
            cv::cvtColor(img, graySrc, cv::COLOR_BGR2GRAY);
        } else {
            graySrc = img;
        }

        boost::ptr_list<CCTag> markers;
        cctag::logtime::Mgmt* durations = nullptr;
        
        // Update CCTag sensitivity parameter
        params._minIdentProba = min_ident_proba;

        try {
            py::gil_scoped_release release;
            cctagDetection(markers, 0, 0, graySrc, params, bank, true, durations);
        } catch (...) {
            return py::list();
        }

        // Returns every marker CCTag found geometrically, not just the ones
        // that reached id_reliable status — including the status code lets
        // the Python side tell apart "nothing was there at all" from "found
        // a ring pattern but rejected it," and *why* (too few outer points,
        // no usable radial cuts, ellipse-refinement didn't converge, or a
        // decoded ID with too little confidence to trust). x()/y()/id()/
        // quality() are all populated regardless of status since they're set
        // during earlier geometric stages, before identification runs.
        // Callers that only want trustworthy detections must filter on
        // status == 1 (status::id_reliable) themselves — this used to be
        // filtered here, silently discarding the diagnostic information.
        py::list results;
        for(const cctag::CCTag& marker : markers) {
            py::dict res;
            res["idx"] = marker.id();
            res["x"] = marker.x() + offset_x;
            res["y"] = marker.y() + offset_y;
            res["decision_margin"] = marker.quality();
            res["status"] = marker.getStatus();
            results.append(res);
        }
        return results;
    }
};

PYBIND11_MODULE(cctag_ext, m) {
    py::class_<FastCCTagDetector>(m, "FastCCTagDetector")
        .def(py::init<int>(), py::arg("nCrowns")=3)
        .def("detect", &FastCCTagDetector::detect, 
             py::arg("input"), py::arg("min_ident_proba")=1e-6f, py::arg("cx")=400.0, py::arg("cy")=300.0, py::arg("fx")=800.0, py::arg("fy")=800.0, py::arg("offset_x")=0.0, py::arg("offset_y")=0.0);
}
