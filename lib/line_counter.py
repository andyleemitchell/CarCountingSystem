import cv2
import numpy as np


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def as_xy_int_tuple(self) -> tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> tuple[float, float]:
        return self.x, self.y


class Vector:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def is_in(self, point: Point) -> bool:
        v1 = Vector(self.start, self.end)
        v2 = Vector(self.start, point)
        cross_product = (v1.end.x - v1.start.x) * (v2.end.y - v2.start.y) - (v1.end.y - v1.start.y) * (
                    v2.end.x - v2.start.x)
        return cross_product < 0


class LineZone:
    """
    Count the number of objects that cross a line.
    """

    def __init__(self, start: Point, end: Point):
        """
        Initialize a LineCounter object.

        Attributes:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.

        """
        self.vector = Vector(start=start, end=end)
        self.tracker_state: dict = {}
        self.in_count: int = 0
        self.out_count: int = 0

    def trigger(self, detections: list):
        """
        Update the in_count and out_count for the detections that cross the line.

        Attributes:
            detections (Detections): The detections for which to update the counts.

        """
        for box, tracker_id in detections:
            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if all four anchors of bbox are on the same side of vector
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]

            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            # detection is partially in and partially out
            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            # handle detection on the same side of the line
            if self.tracker_state[tracker_id] == tracker_state:
                continue

            # updating tracker state
            self.tracker_state[tracker_id] = tracker_state

            # updating the count
            if tracker_state:
                self.in_count += 1
            else:
                self.out_count += 1


class LineZoneAnnotator:
    def __init__(
            self,
            thickness: float = 2,
            color: tuple = (0, 0, 255),
            text_thickness: float = 2,
            text_color: tuple = (0, 0, 0),
            text_scale: float = 0.5,
            text_offset: float = 1.5,
            text_padding: int = 10,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        Attributes:
            thickness (float): The thickness of the line that will be drawn.
            color (Color): The color of the line that will be drawn.
            text_thickness (float): The thickness of the text that will be drawn.
            text_color (Color): The color of the text that will be drawn.
            text_scale (float): The scale of the text that will be drawn.
            text_offset (float): The offset of the text that will be drawn.
            text_padding (int): The padding of the text that will be drawn.

        """
        self.thickness: float = thickness
        self.color: tuple = color
        self.text_thickness: float = text_thickness
        self.text_color: tuple = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, line_counter: LineZone) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        Attributes:
            frame (np.ndarray): The image on which the line will be drawn.
            line_counter (LineCounter): The line counter that will be used to draw the line.

        Returns:
            np.ndarray: The image with the line drawn on it.

        """
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color,
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=5,
            color=self.text_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_counter.vector.end.as_xy_int_tuple(),
            radius=5,
            color=self.text_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        in_text = f"in: {line_counter.in_count}"
        out_text = f"out: {line_counter.out_count}"

        (in_text_width, in_text_height), _ = cv2.getTextSize(
            in_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        (out_text_width, out_text_height), _ = cv2.getTextSize(
            out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        in_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - in_text_width)
            / 2
        )
        in_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + in_text_height)
            / 2
            - self.text_offset * in_text_height
        )

        out_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - out_text_width)
            / 2
        )
        out_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + out_text_height)
            / 2
            + self.text_offset * out_text_height
        )

        cv2.putText(
            frame,
            in_text,
            (in_text_x, in_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color,
            self.text_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            out_text,
            (out_text_x, out_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color,
            self.text_thickness,
            cv2.LINE_AA,
        )
