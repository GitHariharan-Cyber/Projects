<?php
// Log errors but do not show them to users
error_reporting(E_ALL);
ini_set('display_errors', 0);

session_start();

// Only allow access via POST
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405); // Method Not Allowed
    exit("<h1>Method Not Allowed</h1>");
}

// Sanitize POST input
$username = htmlspecialchars($_POST["u"] ?? ''); //Sanitize the input to prevent xss 
$password = htmlspecialchars($_POST["p"] ?? '');

// Use SQLite3 with prepared statements
$con = new SQLite3("app.db");

// Only allow reset by hardcoded admin
$stmt = $con->prepare('SELECT password FROM users WHERE name = :name AND password = :password');
$stmt->bindValue(':name', "admin", SQLITE3_TEXT);  // Force check for admin only
$stmt->bindValue(':password', $password, SQLITE3_TEXT);  // Bind user-supplied password
$result = $stmt->execute();

if ($result->fetchArray()) {
    // Auth successful
    $_SESSION['login'] = true;
    $_SESSION['username'] = 'admin';

    // Reset database
    if (!copy('app_org.db', 'app.db')) {
        error_log("Database copy failed.");
        exit("Internal error during reset.");
    }

    // Reset comments
    $file_pointer = "comments.txt";
    if (file_exists($file_pointer)) {
        unlink($file_pointer);
    }

    // Recreate empty comments file
    $handle = fopen($file_pointer, 'w') or die('Cannot open file:  ' . $file_pointer);
    fclose($handle);

    echo "<h2>The web app has been securely reset.</h2>";
} else {
    echo "<h1>Unauthorized</h1>";
}
?>
