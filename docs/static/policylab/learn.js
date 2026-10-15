(function () {
    'use strict';
    var links = Array.from(document.querySelectorAll('[data-algorithm]'));
    var lessons = Array.from(document.querySelectorAll('[data-lesson]'));
    function show(key) {
        if (!lessons.some(function (lesson) { return lesson.dataset.lesson === key; })) return;
        lessons.forEach(function (lesson) { lesson.hidden = lesson.dataset.lesson !== key; });
        links.forEach(function (link) {
            if (link.dataset.algorithm === key) link.setAttribute('aria-current', 'true');
            else link.removeAttribute('aria-current');
        });
    }
    // Without JavaScript every lesson stays readable and anchor links still work.
    var initial = location.hash.slice(1);
    show(lessons.some(function (lesson) { return lesson.dataset.lesson === initial; }) ? initial : 'bc');
    links.forEach(function (link) {
        link.addEventListener('click', function () { show(link.dataset.algorithm); });
    });
    window.addEventListener('hashchange', function () { show(location.hash.slice(1)); });
}());
