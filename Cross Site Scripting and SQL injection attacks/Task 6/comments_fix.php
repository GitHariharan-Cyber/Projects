<?php 
// DO NOT DISPLAY ERRORS TO USERS
error_reporting(E_ALL); // Track all types of errors internally
ini_set('display_errors', 0);  // But don't show them to the user (for production security)

$pageStart = '<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js"></script>';
$pageStart .= '
<script type="text/javascript">
function checkValid(a){
    if (/[^a-zA-Z0-9 ]/.test(a.value)) {
        $("#helpMessage").text("Error: invalid characters detected");
        return false;
    } else {
        $("#helpMessage").text("");
        return true;
    }
}
</script>';

session_start();

// Session check
if (!isset($_SESSION['login'])) {
    echo "<h2>You must be logged in to access the comments.</h2>";
    exit;
}

function read_comments() {
    $comments = file_get_contents('comments.txt');
    if (empty($comments)) {
        echo '<p><i>There are no comments at this time.</i></p>';
    } else {
        echo $comments;
    }
}

function print_form($val) {
    $sign = hash('sha256', $_SESSION['username']);

    echo '<p><i>Leave a question/comment:</i></p>';
    echo '<span id="helpMessage" class="help-block" style="color:red;"></span>';

    $form = '<form method="post" onsubmit="return checkValid(comment)" action="#">';
    $form .= '<textarea name="comment" id="comment" style="';
    if ($val == 1 || $val == 3) $form .= 'border: 1px solid #912;';
    $form .= 'width: 80%; height: 10em;">';
    if ($val != 0 && isset($_POST['comment'])) {
        $form .= htmlspecialchars($_POST['comment']);
    }
    $form .= '</textarea>';

    $form .= '<p><i>Your name: </i><b>' . htmlspecialchars($_SESSION['username']) . '</b>'; //Sanitized the input username
    $form .= '<input type="hidden" name="token256" value="' . $sign . '"></p>';
    $form .= '<p><input type="submit" value="Post comment"></p>';
    $form .= '</form>';

    $form .= '<form method="post"><p><input type="submit" value="Log me out" name="logout"></p></form>';

    echo $form;
}

function process_form() {
    $err = 0;

    // Handle logout
    if (isset($_POST["logout"])) {
        session_destroy();
        echo "<script>window.location = 'index.php';</script>";
        exit;
    }

    // Only process if comment was submitted
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['comment']) && isset($_POST['token256'])) {
        $comment = trim($_POST["comment"]);
        $token256 = $_POST["token256"];
        $sign = hash('sha256', $_SESSION['username']);

        if (empty($comment)) {
            echo "<p style='color:red;'>Comment cannot be empty.</p>";
            $err = 1;
        }

        if ($token256 !== $sign) {
            echo "<p style='color:red;'>Security token mismatch.</p>";
            $err += 4;
        }

        if ($err === 0) {
            $safeComment = htmlspecialchars($comment); //Used htmlspecialchars to sanitize the comment section
            $fullComment = '<p>“' . $safeComment . '”<br><span style="text-align: right; font-size: 0.75em;">—' . 
                           htmlspecialchars($_SESSION['username']) . ', ' . date('F j, g:i A') . '</span></p>';
            file_put_contents('comments.txt', $fullComment, FILE_APPEND | LOCK_EX);
        } else {
            echo "<p>Error Code: $err</p>";
        }
    }

    read_comments();
    print_form($err);
}

print $pageStart;
process_form();
?>
