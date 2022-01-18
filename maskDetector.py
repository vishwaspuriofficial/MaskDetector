#Title: MASK DETECTOR
#Developer: Vishwas Puri
#Purpose: A program that live detects if you are wearing a mask or not!

#It uses OpenCV and image analysis to detect the human's face and mainly the characteristics of a nose and mouth using machine learning models and uses the information to predict if a user is wearing a mask or not.

#This program is made using python supported by streamlit.

import streamlit as st
import cv2
st.set_page_config(layout="wide")
col = st.empty()

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#defining the nose and mouth classifier models
noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
mouthCascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

st.write("Press start to turn on camera and see if you are wearing your mask or not!")

def maskDetector():
    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            #converting image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #detecting nose in the image
            nose = noseCascade.detectMultiScale(gray, 1.3, 5)
            # detecting mouth in the image
            mouth = mouthCascade.detectMultiScale(gray, 1.3, 5)
            cv2.rectangle(img, pt1=(0, 0), pt2=(700, 80), color=(0, 0, 0), thickness=-1)
            if len(nose) != 0 and len(mouth) != 0:
                #writing "Not Wearing Mask" on image
                cv2.putText(img, "Not Wearing Mask", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            else:
                # writing "Good Wearing Mask!" on image
                cv2.putText(img, "Good, Wearing Mask!", (20, 50), cv2.FONT_HERSHEY_DUPLEX,1, 255)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # setting up streamlit camera configuration
    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "style": {"margin": "0 auto", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )

    # Info Block
    st.write("If camera doesn't turn on, please ensure that your camera permissions are on!")
    with st.expander("Steps to enable permission"):
        st.write("1. Click the lock button at the top left of the page")
        st.write("2. Slide the camera slider to on")
        st.write("3. Reload your page!")


if __name__ == "__main__":
    maskDetector()
