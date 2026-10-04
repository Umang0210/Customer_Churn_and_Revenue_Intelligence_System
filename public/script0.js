
(function() {
  var paths = [
    'dashboard_data.js',
    '/dashboard_data.js',
    '/static/dashboard_data.js'
  ];
  function tryNext(i) {
    if (i >= paths.length) return;
    var s = document.createElement('script');
    s.src = paths[i];
    s.onerror = function() { tryNext(i + 1); };
    document.head.appendChild(s);
  }
  tryNext(0);
})();
