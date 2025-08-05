<?php
// Suppress error display in production, log errors instead
error_reporting(E_ALL);  // Enable reporting of all types of errors internally
ini_set('display_errors', 0); // But don’t show the errors to the browser (security best practice)

echo '<i>Lookup page views</i>';
echo '<form method="get"><input type="text" name="q" id="q" required>';
echo '<p><input type="submit" value="Search"></p></form>';

$q = "";

// If 'q' is set, sanitize it to prevent XSS
if (isset($_GET['q'])) {
    // Sanitize for output (XSS prevention)
    $q = htmlspecialchars($_GET['q'], ENT_QUOTES, 'UTF-8');
}

$con = new SQLite3("app.db");

// Use parameterized query to prevent SQL injection
$stmt = $con->prepare('SELECT views FROM pages WHERE php = :php');
$stmt->bindValue(':php', $q, SQLITE3_TEXT);
$result = $stmt->execute();

$row = $result->fetchArray(SQLITE3_ASSOC);

if ($row) {
    echo "Page <b>$q</b>.php has <i>{$row['views']}</i> views.";
} elseif ($q !== "") {
    echo "This PHP page <b>$q</b> does not exist!";
}
?>
