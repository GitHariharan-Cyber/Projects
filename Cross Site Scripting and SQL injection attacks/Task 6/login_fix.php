<?php
// Production-level error handling
error_reporting(E_ALL);
ini_set('display_errors', 0);

session_start();

// Check that both POST parameters exist
if (!isset($_POST["username"], $_POST["password"])) {
    http_response_code(400);
    exit("<h1>Bad Request</h1>");
}

// Sanitize user input to prevent XSS in output (if echoed)
$username = htmlspecialchars($_POST["username"], ENT_QUOTES, 'UTF-8');
$password = $_POST["password"]; // Don't encode password, needed raw

// Connect to the database
$con = new SQLite3("app.db");

// Use parameterized query to prevent SQL injection
$stmt = $con->prepare('SELECT password FROM users WHERE name = :name AND password = :password');
$stmt->bindValue(':name', $username, SQLITE3_TEXT);
$stmt->bindValue(':password', $password, SQLITE3_TEXT);
$result = $stmt->execute();

// Check if a row was returned
$row = $result->fetchArray(SQLITE3_ASSOC);
if ($row) {
    $_SESSION['login'] = true;
    $_SESSION['username'] = $username;
    header('Location: /comments_TASK7_FIX.php');
    exit();
} else {
    echo "<h1>Login failed.</h1>";
}
?>
