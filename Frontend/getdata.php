<?php
    $conn = mysqli_connect("localhost","creation_creatio","Creation@2018","creation_AQI") or die("Unable to Connect with Database");
    $date = $_GET['date'];
    $value = $_GET['value'];

    $query_insert = "INSERT INTO bda(date,value) VALUES('$date','$value')";
    $result_insert = mysqli_query($conn,$query_insert);
?>