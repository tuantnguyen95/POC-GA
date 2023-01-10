from enum import Enum

APP_NAME = "booster_finding_elements"
ELEMENT_FINDING_URL = "/element_finding"
ELEMENT_COMPARISON_URL = "/element_comparison"
TEXT_ASSERTION_URL = "/text_assertion"
HEALTH_CHECK_URL = "/v1/health-check"
VISUAL_VERIFICATION_URL = "/visual_verification"
FONTSIZE_RECOMMENDATION_URL = "/fontsize"
NAMESPACE = 'ai-service-pid%s-app'
DEVICE_SAME_SIZE_URL = "/device_same_size"
ELEMENT_FINDING_SEGMENTATION_URL = "/element_finding_by_segmentation"
ACCESSIBILITY_ASSERTION_URL = "/accessibility_assertion"

ELEMENT_FINDING_AT_POINT_URL = '/element_finding_at_point'
ELEMENT_FINDING_BY_IMAGE_URL = '/element_finding_by_image'

WBI_URL = '/wbi'

LUNA_ENGINE = "/luna"
FACENET_ENGINE = "/facenet"

RECOMMENDATION_ENGINE = "/recommend"

TENSORFLOW_SERVING_SERVICE = 'tensorflow_serving'
SEGMENTATION_SERVICE = 'segmentation'
KOBITON_OCR_SERVICE = 'ocr'


ANDROID_SCROLLABLE_CLASSES = ('ListView', 'GridView', 'Spinner', 'ScrollView',
                              'AdapterView', 'WebView', 'RecyclerView', 'ViewPager')

IOS_SCROLLABLE_CLASSES = ('XCUIElementTypeCollectionView', 'XCUIElementTypeTable',
                          'XCUIElementTypeWebView', 'XCUIElementTypeScrollView')

WEBVIEW_UNSCROLLABLE_CLASSES = ('nav',)

IOS_TEXT_ATTRS = {
  "XCUIElementTypeAlert": "label",
  "XCUIElementTypeButton": "label",
  "XCUIElementTypeStaticText": "value",
  "XCUIElementTypeTextView": "value",
  "XCUIElementTypeOther": "value",
  "XCUIElementTypeTextField": "value"
}

EDITABLE_ELEMENTS = ('input', 'textarea', 'XCUIElementTypeTextField', 'EditText')
TEXT_ELEMENTS = ('XCUIElementTypeButton', 'XCUIElementTypeStaticText', 'TextView', 'Button')
TOOLBAR = ('XCUIElementTypeToolbar',)

CUT_OFF_CHARS = ['|', '.']

GOOGLE_VISION_IMAGE_PADDING = 3
SOTA_IMAGE_PADDING = 4

MAX_IOU_THRESHOLD = 0.9
INTERSECTION_THRESHOLD = 0.5
REMOVE_LARGE_ELEMENT_ON_SCREEN_THRESHOLD = 2
STITCHING_PADDING_THRESHOLD = 36000

ACCESSIBILITY_ASSERTION_MESSAGE_DICT = {
  'LOW_CONTRAST': 'Low contrast',
  'TOUCH_TARGET_SIZE': 'Element size is small',
  'TOUCH_TARGET_WIDTH': 'Element width is small',
  'TOUCH_TARGET_HEIGHT': 'Element height is small',
}

LUNA_FINDING_ELEMENT_CONFIDENCE_THRESHOLD = 0.55
FACENET_FINDING_ELEMENT_CONFIDENCE_THRESHOLD = 0.7
FINDING_ELEMENT_CONFIDENCE_THRESHOLD = 0.8
FINDING_ELEMENT_CONFIDENCE_THRESHOLD_SCROLL = 0.9

EXCEPT_RESOLUTION_IPHONES = ['iPhone 6 Plus', 'iPhone 6s Plus', 'iPhone 7 Plus', 'iPhone 8 Plus']

DENSITY_BUCKET_DICT = {'MDPI': {'scale': 1, 'dpi': 160}, 'HDPI': {'scale': 1.5, 'dpi': 240},
                  'XHDPI': {'scale': 2, 'dpi': 320}, 'XXHDPI': {'scale': 3, 'dpi': 480},
                  'XXXHDPI': {'scale': 4, 'dpi': 640}}

DENSITY_BUCKETS = ['MDPI', 'XHDPI', 'XXXHDPI', 'HDPI', 'XXHDPI']

ELEMENT_TYPES = ['android.widget.HorizontalScrollView', 'androidx.cardview.widget.CardView', 'android.widget.ListView',
                 'android.widget.Image', 'android.widget.EditText', 'android.webkit.WebView', 'android.view.View',
                 'android.widget.Button', 'com.android.launcher3.home.Workspace', 'android.widget.ProgressBar',
                 'android.widget.ScrollView', 'android.widget.CheckedTextView', 'android.widget.LinearLayout',
                 'android.widget.ImageButton', 'androidx.viewpager.widget.ViewPager', 'android.view.ViewGroup',
                 'android.widget.RelativeLayout', 'android.widget.TextView',
                 'androidx.recyclerview.widget.RecyclerView', 'android.widget.Switch', 'android.widget.CheckBox',
                 'android.widget.ImageView', 'android.widget.ToggleButton']

SCREEN_PARTS = ['body', 'scroll', 'header', 'footer']
KEYBOARD_PACKAGE = ['com.sec.android.inputmethod', 'com.google.android.inputmethod.latin', 'com.samsung.android.honeyboard']

class Platform(Enum):
  ANDROID, IOS = range(2)


class TextAssertion(Enum):
  EXACT, BEGINNING_OF_TEXT, RELAXED_PLACEMENT, CONTAIN, SKIP = range(5)


class TextAssertionStatus(Enum):
  PASSED, FAILED = range(2)


###############################################
# <= 1.0: Not perceptible by the human eye
# 1-2: Perceptible through close observation
# 2-10: Perceptible at a glance
# 11-49: Colors are more similar than the opposite
# 100: Colors are exactly the opposite
###############################################
class ColorTextAssertion(Enum):
  STRICT, RELAXED, SKIP = [10., 49., 0.]


class StructureAssertionStatus(Enum):
  PASSED, FAILED, SKIP = range(3)


class LayoutAssertionStatus(Enum):
  PASSED, FAILED, SKIP = range(3)


class FontSizeAssertionStatus(Enum):
  PASSED, FAILED, SKIP = range(3)


class StructureAssertion(Enum):
  STRICT, SKIP = range(2)


class LayoutAssertion(Enum):
  STRICT, RELAXED_PLACEMENT, SKIP = range(3)


class FontSizeStatus(Enum):
  SMALL, GOOD, LARGE, SKIP = range(4)


class LocationInScreen(Enum):
  HEADER, BODY, FOOTER, SCROLL = range(4)


class ElementType(Enum):
  SCROLLABLE, TEXTUAL, VISUAL, KEYBOARD = range(4)


class AccessibilityAssertionType(Enum):
  ERROR, WARNING = range(2)


class AccessibilityAssertionCategory(Enum):
  LOW_CONTRAST, TOUCH_TARGET_SIZE, TOUCH_TARGET_WIDTH, TOUCH_TARGET_HEIGHT = range(4)


ELEMENT_SELECTION_MODEL_PATH = 'service/element_selection_model/model_v3.txt'

WBI_MODEL_PATH = 'service/wbi/wbi_svc.pkl'


def get_scrollable_classes(platform):
  if platform == Platform.ANDROID:
    return ANDROID_SCROLLABLE_CLASSES
  return IOS_SCROLLABLE_CLASSES


TOUCH_PADDING = 15
# For Luna matching
MARKER_GAP = 0.2
MIN_AMBIGUOUS_THRESHOLD = 0.1
MIN_AMBIGUOUS_GAP = 0.02
MAX_MIN_THRESHOLD_GAP = 0.5
MARKER_PADDING = 0.05
NUMBER_MARKERS_NEED = 3  # 3 nonlinear points make a plane
DEFAULT_GAP_THRESHOLD = 0.2
AREA_PADDING = 1.1
