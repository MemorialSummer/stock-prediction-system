function toggleDetails(detailsId) {
    var details = document.getElementById(detailsId);
    if (details.style.display === 'none') {
        details.style.display = 'block';
    } else {
        details.style.display = 'none';
    }
}