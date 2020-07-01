/*! For license information please see d93723539f32b8051de9.worker.js.LICENSE.txt */
!function(t){var r={};function e(n){if(r[n])return r[n].exports;var o=r[n]={i:n,l:!1,exports:{}};return t[n].call(o.exports,o,o.exports,e),o.l=!0,o.exports}e.m=t,e.c=r,e.d=function(t,r,n){e.o(t,r)||Object.defineProperty(t,r,{enumerable:!0,get:n})},e.r=function(t){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},e.t=function(t,r){if(1&r&&(t=e(t)),8&r)return t;if(4&r&&"object"===typeof t&&t&&t.__esModule)return t;var n=Object.create(null);if(e.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:t}),2&r&&"string"!=typeof t)for(var o in t)e.d(n,o,function(r){return t[r]}.bind(null,o));return n},e.n=function(t){var r=t&&t.__esModule?function(){return t.default}:function(){return t};return e.d(r,"a",r),r},e.o=function(t,r){return Object.prototype.hasOwnProperty.call(t,r)},e.p="./",e(e.s=12)}([function(t,r,e){"use strict";var n=e(5),o=e(8);function i(){this.protocol=null,this.slashes=null,this.auth=null,this.host=null,this.port=null,this.hostname=null,this.hash=null,this.search=null,this.query=null,this.pathname=null,this.path=null,this.href=null}r.parse=g,r.resolve=function(t,r){return g(t,!1,!0).resolve(r)},r.resolveObject=function(t,r){return t?g(t,!1,!0).resolveObject(r):r},r.format=function(t){o.isString(t)&&(t=g(t));return t instanceof i?t.format():i.prototype.format.call(t)},r.Url=i;var s=/^([a-z0-9.+-]+:)/i,a=/:[0-9]*$/,u=/^(\/\/?(?!\/)[^\?\s]*)(\?[^\s]*)?$/,h=["{","}","|","\\","^","`"].concat(["<",">",'"',"`"," ","\r","\n","\t"]),c=["'"].concat(h),f=["%","/","?",";","#"].concat(c),l=["/","?","#"],p=/^[+a-z0-9A-Z_-]{0,63}$/,m=/^([+a-z0-9A-Z_-]{0,63})(.*)$/,d={javascript:!0,"javascript:":!0},v={javascript:!0,"javascript:":!0},y={http:!0,https:!0,ftp:!0,gopher:!0,file:!0,"http:":!0,"https:":!0,"ftp:":!0,"gopher:":!0,"file:":!0},w=e(9);function g(t,r,e){if(t&&o.isObject(t)&&t instanceof i)return t;var n=new i;return n.parse(t,r,e),n}i.prototype.parse=function(t,r,e){if(!o.isString(t))throw new TypeError("Parameter 'url' must be a string, not "+typeof t);var i=t.indexOf("?"),a=-1!==i&&i<t.indexOf("#")?"?":"#",h=t.split(a);h[0]=h[0].replace(/\\/g,"/");var g=t=h.join(a);if(g=g.trim(),!e&&1===t.split("#").length){var b=u.exec(g);if(b)return this.path=g,this.href=g,this.pathname=b[1],b[2]?(this.search=b[2],this.query=r?w.parse(this.search.substr(1)):this.search.substr(1)):r&&(this.search="",this.query={}),this}var x=s.exec(g);if(x){var O=(x=x[0]).toLowerCase();this.protocol=O,g=g.substr(x.length)}if(e||x||g.match(/^\/\/[^@\/]+@[^@\/]+/)){var _="//"===g.substr(0,2);!_||x&&v[x]||(g=g.substr(2),this.slashes=!0)}if(!v[x]&&(_||x&&!y[x])){for(var k,E,j=-1,A=0;A<l.length;A++){-1!==(S=g.indexOf(l[A]))&&(-1===j||S<j)&&(j=S)}-1!==(E=-1===j?g.lastIndexOf("@"):g.lastIndexOf("@",j))&&(k=g.slice(0,E),g=g.slice(E+1),this.auth=decodeURIComponent(k)),j=-1;for(A=0;A<f.length;A++){var S;-1!==(S=g.indexOf(f[A]))&&(-1===j||S<j)&&(j=S)}-1===j&&(j=g.length),this.host=g.slice(0,j),g=g.slice(j),this.parseHost(),this.hostname=this.hostname||"";var T="["===this.hostname[0]&&"]"===this.hostname[this.hostname.length-1];if(!T)for(var I=this.hostname.split(/\./),M=(A=0,I.length);A<M;A++){var C=I[A];if(C&&!C.match(p)){for(var P="",G=0,q=C.length;G<q;G++)C.charCodeAt(G)>127?P+="x":P+=C[G];if(!P.match(p)){var R=I.slice(0,A),U=I.slice(A+1),z=C.match(m);z&&(R.push(z[1]),U.unshift(z[2])),U.length&&(g="/"+U.join(".")+g),this.hostname=R.join(".");break}}}this.hostname.length>255?this.hostname="":this.hostname=this.hostname.toLowerCase(),T||(this.hostname=n.toASCII(this.hostname));var N=this.port?":"+this.port:"",L=this.hostname||"";this.host=L+N,this.href+=this.host,T&&(this.hostname=this.hostname.substr(1,this.hostname.length-2),"/"!==g[0]&&(g="/"+g))}if(!d[O])for(A=0,M=c.length;A<M;A++){var F=c[A];if(-1!==g.indexOf(F)){var B=encodeURIComponent(F);B===F&&(B=escape(F)),g=g.split(F).join(B)}}var W=g.indexOf("#");-1!==W&&(this.hash=g.substr(W),g=g.slice(0,W));var D=g.indexOf("?");if(-1!==D?(this.search=g.substr(D),this.query=g.substr(D+1),r&&(this.query=w.parse(this.query)),g=g.slice(0,D)):r&&(this.search="",this.query={}),g&&(this.pathname=g),y[O]&&this.hostname&&!this.pathname&&(this.pathname="/"),this.pathname||this.search){N=this.pathname||"";var V=this.search||"";this.path=N+V}return this.href=this.format(),this},i.prototype.format=function(){var t=this.auth||"";t&&(t=(t=encodeURIComponent(t)).replace(/%3A/i,":"),t+="@");var r=this.protocol||"",e=this.pathname||"",n=this.hash||"",i=!1,s="";this.host?i=t+this.host:this.hostname&&(i=t+(-1===this.hostname.indexOf(":")?this.hostname:"["+this.hostname+"]"),this.port&&(i+=":"+this.port)),this.query&&o.isObject(this.query)&&Object.keys(this.query).length&&(s=w.stringify(this.query));var a=this.search||s&&"?"+s||"";return r&&":"!==r.substr(-1)&&(r+=":"),this.slashes||(!r||y[r])&&!1!==i?(i="//"+(i||""),e&&"/"!==e.charAt(0)&&(e="/"+e)):i||(i=""),n&&"#"!==n.charAt(0)&&(n="#"+n),a&&"?"!==a.charAt(0)&&(a="?"+a),r+i+(e=e.replace(/[?#]/g,(function(t){return encodeURIComponent(t)})))+(a=a.replace("#","%23"))+n},i.prototype.resolve=function(t){return this.resolveObject(g(t,!1,!0)).format()},i.prototype.resolveObject=function(t){if(o.isString(t)){var r=new i;r.parse(t,!1,!0),t=r}for(var e=new i,n=Object.keys(this),s=0;s<n.length;s++){var a=n[s];e[a]=this[a]}if(e.hash=t.hash,""===t.href)return e.href=e.format(),e;if(t.slashes&&!t.protocol){for(var u=Object.keys(t),h=0;h<u.length;h++){var c=u[h];"protocol"!==c&&(e[c]=t[c])}return y[e.protocol]&&e.hostname&&!e.pathname&&(e.path=e.pathname="/"),e.href=e.format(),e}if(t.protocol&&t.protocol!==e.protocol){if(!y[t.protocol]){for(var f=Object.keys(t),l=0;l<f.length;l++){var p=f[l];e[p]=t[p]}return e.href=e.format(),e}if(e.protocol=t.protocol,t.host||v[t.protocol])e.pathname=t.pathname;else{for(var m=(t.pathname||"").split("/");m.length&&!(t.host=m.shift()););t.host||(t.host=""),t.hostname||(t.hostname=""),""!==m[0]&&m.unshift(""),m.length<2&&m.unshift(""),e.pathname=m.join("/")}if(e.search=t.search,e.query=t.query,e.host=t.host||"",e.auth=t.auth,e.hostname=t.hostname||t.host,e.port=t.port,e.pathname||e.search){var d=e.pathname||"",w=e.search||"";e.path=d+w}return e.slashes=e.slashes||t.slashes,e.href=e.format(),e}var g=e.pathname&&"/"===e.pathname.charAt(0),b=t.host||t.pathname&&"/"===t.pathname.charAt(0),x=b||g||e.host&&t.pathname,O=x,_=e.pathname&&e.pathname.split("/")||[],k=(m=t.pathname&&t.pathname.split("/")||[],e.protocol&&!y[e.protocol]);if(k&&(e.hostname="",e.port=null,e.host&&(""===_[0]?_[0]=e.host:_.unshift(e.host)),e.host="",t.protocol&&(t.hostname=null,t.port=null,t.host&&(""===m[0]?m[0]=t.host:m.unshift(t.host)),t.host=null),x=x&&(""===m[0]||""===_[0])),b)e.host=t.host||""===t.host?t.host:e.host,e.hostname=t.hostname||""===t.hostname?t.hostname:e.hostname,e.search=t.search,e.query=t.query,_=m;else if(m.length)_||(_=[]),_.pop(),_=_.concat(m),e.search=t.search,e.query=t.query;else if(!o.isNullOrUndefined(t.search)){if(k)e.hostname=e.host=_.shift(),(T=!!(e.host&&e.host.indexOf("@")>0)&&e.host.split("@"))&&(e.auth=T.shift(),e.host=e.hostname=T.shift());return e.search=t.search,e.query=t.query,o.isNull(e.pathname)&&o.isNull(e.search)||(e.path=(e.pathname?e.pathname:"")+(e.search?e.search:"")),e.href=e.format(),e}if(!_.length)return e.pathname=null,e.search?e.path="/"+e.search:e.path=null,e.href=e.format(),e;for(var E=_.slice(-1)[0],j=(e.host||t.host||_.length>1)&&("."===E||".."===E)||""===E,A=0,S=_.length;S>=0;S--)"."===(E=_[S])?_.splice(S,1):".."===E?(_.splice(S,1),A++):A&&(_.splice(S,1),A--);if(!x&&!O)for(;A--;A)_.unshift("..");!x||""===_[0]||_[0]&&"/"===_[0].charAt(0)||_.unshift(""),j&&"/"!==_.join("/").substr(-1)&&_.push("");var T,I=""===_[0]||_[0]&&"/"===_[0].charAt(0);k&&(e.hostname=e.host=I?"":_.length?_.shift():"",(T=!!(e.host&&e.host.indexOf("@")>0)&&e.host.split("@"))&&(e.auth=T.shift(),e.host=e.hostname=T.shift()));return(x=x||e.host&&_.length)&&!I&&_.unshift(""),_.length?e.pathname=_.join("/"):(e.pathname=null,e.path=null),o.isNull(e.pathname)&&o.isNull(e.search)||(e.path=(e.pathname?e.pathname:"")+(e.search?e.search:"")),e.auth=t.auth||e.auth,e.slashes=e.slashes||t.slashes,e.href=e.format(),e},i.prototype.parseHost=function(){var t=this.host,r=a.exec(t);r&&(":"!==(r=r[0])&&(this.port=r.substr(1)),t=t.substr(0,t.length-r.length)),t&&(this.hostname=t)}},function(t,r,e){"use strict";var n=this&&this.__importDefault||function(t){return t&&t.__esModule?t:{default:t}};Object.defineProperty(r,"__esModule",{value:!0});var o=n(e(2)),i=n(e(3));function s(t){for(var r=[],e=0;e<t.length;e++){var n=t[e];if(!n.isSome())return o.default.none();r.push(n.value)}return o.default.some(r)}function a(t){for(var r=[],e=0;e<t.length;e++){var n=t[e];if(!n.isOk())return n;r.push(n.safeUnwrap())}return i.default.ok(r)}r.option={some:function(t){return o.default.some(t)},none:function(){return o.default.none()},all:s,fromVoidable:function(t){return void 0===t||null===t?o.default.none():o.default.some(t)}},r.optionDotAll=s,r.result={ok:function(t){return i.default.ok(t)},err:function(t){return i.default.err(t)},all:a},r.resultDotAll=a},function(t,r,e){"use strict";var n=this&&this.__importDefault||function(t){return t&&t.__esModule?t:{default:t}};Object.defineProperty(r,"__esModule",{value:!0});var o=n(e(3)),i=n(e(4)),s=function(){function t(t,r){this.isNone_=t,this.value=r}return t.some=function(r){return new t(!1,r)},t.none=function(){return new t(!0,void 0)},t.prototype.match=function(t){return this.isNone()?t.none():t.some(this.value)},t.prototype.isNone=function(){return this.isNone_},t.prototype.isSome=function(){return!this.isNone()},t.prototype.map=function(r){var e=this;return this.match({none:function(){return e},some:function(e){return t.some(r(e))}})},t.prototype.ifSome=function(t){this.map(t)},t.prototype.ifNone=function(t){this.isNone()&&t()},t.prototype.unwrap=function(){return this.expect("Tried to call unwrap() on option.none()")},t.prototype.expect=function(t){return this.match({none:function(){throw"string"===typeof t?new i.default(t):t},some:function(t){return t}})},t.prototype.unwrapOr=function(t){return this.match({none:function(){return t},some:function(t){return t}})},t.prototype.unwrapOrElse=function(t){return this.match({none:function(){return t()},some:function(t){return t}})},t.prototype.and=function(r){return this.match({none:function(){return t.none()},some:function(){return r}})},t.prototype.andThen=function(r){return this.match({none:function(){return t.none()},some:r})},t.prototype.or=function(t){var r=this;return this.match({none:function(){return t},some:function(){return r}})},t.prototype.orElse=function(t){var r=this;return this.match({none:t,some:function(){return r}})},t.prototype.filter=function(r){var e=this;return this.andThen((function(n){return r(n)?e:t.none()}))},t.prototype.flatten=function(){return this.andThen((function(t){return t}))},t.prototype.array=function(){return this.match({none:function(){return[]},some:function(t){return[t]}})},t.prototype.xor=function(r){var e=this;return this.match({none:function(){return r},some:function(){return r.match({none:function(){return e},some:function(){return t.none()}})}})},t.prototype.transpose=function(){return this.match({none:function(){return o.default.ok(t.none())},some:function(r){return r.match({ok:function(r){return o.default.ok(t.some(r))},err:function(t){return o.default.err(t)}})}})},t.prototype.equalsSome=function(t){return this.match({none:function(){return!1},some:function(r){return r===t}})},t.prototype.someSatisfies=function(t){return this.match({none:function(){return!1},some:t})},t}();r.default=s},function(t,r,e){"use strict";var n=this&&this.__importDefault||function(t){return t&&t.__esModule?t:{default:t}};Object.defineProperty(r,"__esModule",{value:!0});var o=n(e(2)),i=n(e(4)),s=function(){function t(t,r){this.isErr_=t,this.value=r}return t.ok=function(r){return new t(!1,r)},t.err=function(r){return new t(!0,r)},t.prototype.match=function(t){return this.isErr_?t.err(this.value):t.ok(this.value)},t.prototype.ok=function(){return this.match({ok:function(t){return o.default.some(t)},err:function(){return o.default.none()}})},t.prototype.err=function(){return this.match({ok:function(){return o.default.none()},err:function(t){return o.default.some(t)}})},t.prototype.isOk=function(){return!this.isErr_},t.prototype.isErr=function(){return this.isErr_},t.prototype.map=function(r){var e=this;return this.match({ok:function(e){return t.ok(r(e))},err:function(){return e}})},t.prototype.mapErr=function(r){var e=this;return this.match({err:function(e){return t.err(r(e))},ok:function(){return e}})},t.prototype.ifOk=function(t){this.map(t)},t.prototype.ifErr=function(t){this.mapErr(t)},t.prototype.unwrap=function(){return this.expect("Tried to call unwrap() on result.err()")},t.prototype.safeUnwrap=function(){return this.value},t.prototype.safeUnwrapErr=function(){return this.value},t.prototype.unwrapErr=function(){return this.expectErr("Tried to call unwrapErr() on result.ok()")},t.prototype.unwrapOrThrowErr=function(){return this.match({ok:function(t){return t},err:function(t){throw t}})},t.prototype.unwrapErrOrThrowOk=function(){return this.match({ok:function(t){throw t},err:function(t){return t}})},t.prototype.expect=function(t){return this.match({ok:function(t){return t},err:function(){throw"string"===typeof t?new i.default(t):t}})},t.prototype.expectErr=function(t){return this.match({ok:function(){throw"string"===typeof t?new i.default(t):t},err:function(t){return t}})},t.prototype.unwrapOr=function(t){return this.match({ok:function(t){return t},err:function(){return t}})},t.prototype.unwrapOrElse=function(t){var r=this;return this.match({ok:function(t){return t},err:function(){return t(r.value)}})},t.prototype.and=function(t){var r=this;return this.match({ok:function(){return t},err:function(){return r}})},t.prototype.andThen=function(t){var r=this;return this.match({ok:t,err:function(){return r}})},t.prototype.or=function(t){var r=this;return this.match({ok:function(){return r},err:function(){return t}})},t.prototype.orElse=function(t){var r=this;return this.match({ok:function(){return r},err:t})},t.prototype.array=function(){return this.match({ok:function(t){return[t]},err:function(){return[]}})},t.prototype.transpose=function(){return this.match({ok:function(r){return r.match({some:function(r){return o.default.some(t.ok(r))},none:function(){return o.default.none()}})},err:function(r){return o.default.some(t.err(r))}})},t.prototype.okSatisfies=function(t){return this.ok().match({some:t,none:function(){return!1}})},t.prototype.errSatisfies=function(t){return this.err().match({some:t,none:function(){return!1}})},t.prototype.reverse=function(){return this.match({ok:function(r){return t.err(r)},err:function(r){return t.ok(r)}})},t}();r.default=s},function(t,r,e){"use strict";var n=this&&this.__extends||function(){var t=function(r,e){return(t=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(t,r){t.__proto__=r}||function(t,r){for(var e in r)r.hasOwnProperty(e)&&(t[e]=r[e])})(r,e)};return function(r,e){function n(){this.constructor=r}t(r,e),r.prototype=null===e?Object.create(e):(n.prototype=e.prototype,new n)}}();Object.defineProperty(r,"__esModule",{value:!0});var o=function(t){function r(r){var e=t.call(this,r)||this;return e.name="UnwrapError",e}return n(r,t),r}(Error);r.default=o},function(t,r,e){(function(t,n){var o;!function(i){r&&r.nodeType,t&&t.nodeType;var s="object"==typeof n&&n;s.global!==s&&s.window!==s&&s.self;var a,u=2147483647,h=/^xn--/,c=/[^\x20-\x7E]/,f=/[\x2E\u3002\uFF0E\uFF61]/g,l={overflow:"Overflow: input needs wider integers to process","not-basic":"Illegal input >= 0x80 (not a basic code point)","invalid-input":"Invalid input"},p=Math.floor,m=String.fromCharCode;function d(t){throw new RangeError(l[t])}function v(t,r){for(var e=t.length,n=[];e--;)n[e]=r(t[e]);return n}function y(t,r){var e=t.split("@"),n="";return e.length>1&&(n=e[0]+"@",t=e[1]),n+v((t=t.replace(f,".")).split("."),r).join(".")}function w(t){for(var r,e,n=[],o=0,i=t.length;o<i;)(r=t.charCodeAt(o++))>=55296&&r<=56319&&o<i?56320==(64512&(e=t.charCodeAt(o++)))?n.push(((1023&r)<<10)+(1023&e)+65536):(n.push(r),o--):n.push(r);return n}function g(t){return v(t,(function(t){var r="";return t>65535&&(r+=m((t-=65536)>>>10&1023|55296),t=56320|1023&t),r+=m(t)})).join("")}function b(t,r){return t+22+75*(t<26)-((0!=r)<<5)}function x(t,r,e){var n=0;for(t=e?p(t/700):t>>1,t+=p(t/r);t>455;n+=36)t=p(t/35);return p(n+36*t/(t+38))}function O(t){var r,e,n,o,i,s,a,h,c,f,l,m=[],v=t.length,y=0,w=128,b=72;for((e=t.lastIndexOf("-"))<0&&(e=0),n=0;n<e;++n)t.charCodeAt(n)>=128&&d("not-basic"),m.push(t.charCodeAt(n));for(o=e>0?e+1:0;o<v;){for(i=y,s=1,a=36;o>=v&&d("invalid-input"),((h=(l=t.charCodeAt(o++))-48<10?l-22:l-65<26?l-65:l-97<26?l-97:36)>=36||h>p((u-y)/s))&&d("overflow"),y+=h*s,!(h<(c=a<=b?1:a>=b+26?26:a-b));a+=36)s>p(u/(f=36-c))&&d("overflow"),s*=f;b=x(y-i,r=m.length+1,0==i),p(y/r)>u-w&&d("overflow"),w+=p(y/r),y%=r,m.splice(y++,0,w)}return g(m)}function _(t){var r,e,n,o,i,s,a,h,c,f,l,v,y,g,O,_=[];for(v=(t=w(t)).length,r=128,e=0,i=72,s=0;s<v;++s)(l=t[s])<128&&_.push(m(l));for(n=o=_.length,o&&_.push("-");n<v;){for(a=u,s=0;s<v;++s)(l=t[s])>=r&&l<a&&(a=l);for(a-r>p((u-e)/(y=n+1))&&d("overflow"),e+=(a-r)*y,r=a,s=0;s<v;++s)if((l=t[s])<r&&++e>u&&d("overflow"),l==r){for(h=e,c=36;!(h<(f=c<=i?1:c>=i+26?26:c-i));c+=36)O=h-f,g=36-f,_.push(m(b(f+O%g,0))),h=p(O/g);_.push(m(b(h,0))),i=x(e,y,n==o),e=0,++n}++e,++r}return _.join("")}a={version:"1.4.1",ucs2:{decode:w,encode:g},decode:O,encode:_,toASCII:function(t){return y(t,(function(t){return c.test(t)?"xn--"+_(t):t}))},toUnicode:function(t){return y(t,(function(t){return h.test(t)?O(t.slice(4).toLowerCase()):t}))}},void 0===(o=function(){return a}.call(r,e,r,t))||(t.exports=o)}()}).call(this,e(6)(t),e(7))},function(t,r){t.exports=function(t){return t.webpackPolyfill||(t.deprecate=function(){},t.paths=[],t.children||(t.children=[]),Object.defineProperty(t,"loaded",{enumerable:!0,get:function(){return t.l}}),Object.defineProperty(t,"id",{enumerable:!0,get:function(){return t.i}}),t.webpackPolyfill=1),t}},function(t,r){var e;e=function(){return this}();try{e=e||new Function("return this")()}catch(n){"object"===typeof window&&(e=window)}t.exports=e},function(t,r,e){"use strict";t.exports={isString:function(t){return"string"===typeof t},isObject:function(t){return"object"===typeof t&&null!==t},isNull:function(t){return null===t},isNullOrUndefined:function(t){return null==t}}},function(t,r,e){"use strict";r.decode=r.parse=e(10),r.encode=r.stringify=e(11)},function(t,r,e){"use strict";function n(t,r){return Object.prototype.hasOwnProperty.call(t,r)}t.exports=function(t,r,e,i){r=r||"&",e=e||"=";var s={};if("string"!==typeof t||0===t.length)return s;var a=/\+/g;t=t.split(r);var u=1e3;i&&"number"===typeof i.maxKeys&&(u=i.maxKeys);var h=t.length;u>0&&h>u&&(h=u);for(var c=0;c<h;++c){var f,l,p,m,d=t[c].replace(a,"%20"),v=d.indexOf(e);v>=0?(f=d.substr(0,v),l=d.substr(v+1)):(f=d,l=""),p=decodeURIComponent(f),m=decodeURIComponent(l),n(s,p)?o(s[p])?s[p].push(m):s[p]=[s[p],m]:s[p]=m}return s};var o=Array.isArray||function(t){return"[object Array]"===Object.prototype.toString.call(t)}},function(t,r,e){"use strict";var n=function(t){switch(typeof t){case"string":return t;case"boolean":return t?"true":"false";case"number":return isFinite(t)?t:"";default:return""}};t.exports=function(t,r,e,a){return r=r||"&",e=e||"=",null===t&&(t=void 0),"object"===typeof t?i(s(t),(function(s){var a=encodeURIComponent(n(s))+e;return o(t[s])?i(t[s],(function(t){return a+encodeURIComponent(n(t))})).join(r):a+encodeURIComponent(n(t[s]))})).join(r):a?encodeURIComponent(n(a))+e+encodeURIComponent(n(t)):""};var o=Array.isArray||function(t){return"[object Array]"===Object.prototype.toString.call(t)};function i(t,r){if(t.map)return t.map(r);for(var e=[],n=0;n<t.length;n++)e.push(r(t[n],n));return e}var s=Object.keys||function(t){var r=[];for(var e in t)Object.prototype.hasOwnProperty.call(t,e)&&r.push(e);return r}},function(t,r,e){"use strict";function n(t,r,e){return r in t?Object.defineProperty(t,r,{value:e,enumerable:!0,configurable:!0,writable:!0}):t[r]=e,t}function o(t,r){var e=Object.keys(t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(t);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(t,r).enumerable}))),e.push.apply(e,n)}return e}function i(t){for(var r=1;r<arguments.length;r++){var e=null!=arguments[r]?arguments[r]:{};r%2?o(Object(e),!0).forEach((function(r){n(t,r,e[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(e)):o(Object(e)).forEach((function(r){Object.defineProperty(t,r,Object.getOwnPropertyDescriptor(e,r))}))}return t}function s(t,r){(null==r||r>t.length)&&(r=t.length);for(var e=0,n=new Array(r);e<r;e++)n[e]=t[e];return n}function a(t,r){if(t){if("string"===typeof t)return s(t,r);var e=Object.prototype.toString.call(t).slice(8,-1);return"Object"===e&&t.constructor&&(e=t.constructor.name),"Map"===e||"Set"===e?Array.from(e):"Arguments"===e||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(e)?s(t,r):void 0}}function u(t){return function(t){if(Array.isArray(t))return s(t)}(t)||function(t){if("undefined"!==typeof Symbol&&Symbol.iterator in Object(t))return Array.from(t)}(t)||a(t)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function h(t,r){if(!(t instanceof r))throw new TypeError("Cannot call a class as a function")}function c(t,r){for(var e=0;e<r.length;e++){var n=r[e];n.enumerable=n.enumerable||!1,n.configurable=!0,"value"in n&&(n.writable=!0),Object.defineProperty(t,n.key,n)}}function f(t,r,e){return r&&c(t.prototype,r),e&&c(t,e),t}e.r(r);var l=function(){function t(r,e,n){h(this,t),this.rows=void 0,this.columns=void 0,this.data=void 0,this.rows=r,this.columns=e,this.data=n instanceof Float64Array?n:Float64Array.from(n)}return f(t,null,[{key:"randomUniform",value:function(r,e){for(var n=r*e,o=new Float64Array(n),i=0;i<n;i++)o[i]=2*Math.random()-1;return new t(r,e,o)}},{key:"fromEntryInitializer",value:function(r,e,n){return new t(r,e,new Float64Array(r*e).map(n))}},{key:"zeros",value:function(r,e){return new t(r,e,new Float64Array(r*e))}},{key:"fromRows",value:function(r){var e=r[0].length;if(r.some((function(t){return t.length!==e})))throw new Error("Cannot create a matrix from a jagged array: "+JSON.stringify(r));return new t(r.length,e,r.flat())}},{key:"columnVector",value:function(r){return new t(r.length,1,r)}},{key:"fromRowMajorOrderEntries",value:function(r,e,n){if(n.length!==r*e)throw new Error("Expected "+r*e+" entries but instead got "+n.length+".");return new t(r,e,n)}}]),f(t,[{key:"clone",value:function(){return new t(this.rows,this.columns,this.data.slice())}},{key:"mutMultiplyScalar",value:function(t){for(var r=this.data.length,e=0;e<r;e++)this.data[e]*=t;return this}},{key:"multiplyScalarInto",value:function(t,r){if(this.rows!==r.rows||this.columns!==r.columns)throw new Error("Cannot multiply a scalar "+t+" by a "+this.rows+"x"+this.columns+" matrix into a "+r.rows+"x"+r.columns+" matrix. The out matrix must have the same dimensions as this matrix.");for(var e=this.data,n=r.data,o=n.length,i=0;i<o;i++)n[i]=t*e[i];return r}},{key:"mutAdd",value:function(t){if(t.rows!==this.rows||t.columns!==this.columns)throw new TypeError("Cannot add a "+this.rows+"x"+this.columns+" to a "+t.rows+"x"+t.columns+" matrix.");for(var r=this.data.length,e=0;e<r;e++)this.data[e]+=t.data[e];return this}},{key:"mutSubtract",value:function(t){if(t.rows!==this.rows||t.columns!==this.columns)throw new TypeError("Cannot add a "+this.rows+"x"+this.columns+" to a "+t.rows+"x"+t.columns+" matrix.");for(var r=this.data.length,e=0;e<r;e++)this.data[e]-=t.data[e];return this}},{key:"immutSubtract",value:function(t){return this.subtractInto(t,this.clone())}},{key:"subtractInto",value:function(t,r){if(t.rows!==this.rows||t.columns!==this.columns)throw new TypeError("Cannot add a "+this.rows+"x"+this.columns+" matrix to a "+t.rows+"x"+t.columns+" matrix.");for(var e=this.data,n=t.data,o=r.data,i=o.length,s=0;s<i;s++)o[s]=e[s]-n[s];return r}},{key:"immutMultiply",value:function(r){return this.multiplyInto(r,t.zeros(this.rows,r.columns))}},{key:"multiplyInto",value:function(t,r){if(this.columns!==t.rows)throw new TypeError("Cannot multiply a "+this.rows+"x"+this.columns+" matrix with a "+t.rows+"x"+t.columns+" matrix.");if(this.rows!==r.rows||t.columns!==r.columns)throw new TypeError("Cannot multiply a "+this.rows+"x"+this.columns+" matrix with a "+t.rows+"x"+t.columns+" matrix into a "+r.rows+"x"+r.columns+" matrix.");for(var e=this.data,n=t.data,o=r.data,i=this.rows,s=t.columns,a=this.columns,u=r.columns,h=0;h<i;h++)for(var c=0;c<s;c++){for(var f=0,l=0;l<a;l++)f+=e[h*a+l]*n[l*s+c];o[h*u+c]=f}return r}},{key:"mutHadamard",value:function(t){if(t.rows!==this.rows||t.columns!==this.columns)throw new TypeError("Cannot take the Hadamard product of a "+this.rows+"x"+this.columns+" matrix and a "+t.rows+"x"+t.columns+" matrix.");for(var r=this.data.length,e=0;e<r;e++)this.data[e]*=t.data[e];return this}},{key:"immutTranspose",value:function(){return this.transposeInto(new t(this.columns,this.rows,new Float64Array(this.data.length)))}},{key:"transposeInto",value:function(t){if(this.rows!==t.columns||this.columns!==t.rows)throw new Error("Cannot transpose a "+this.rows+"x"+this.columns+" matrix into a "+t.rows+"x"+t.columns+" matrix.");for(var r=this.data,e=this.rows,n=this.columns,o=t.data,i=t.columns,s=0;s<e;s++)for(var a=0;a<n;a++)o[a*i+s]=r[s*n+a];return t}},{key:"rowMajorOrderEntries",value:function(){return this.data}},{key:"immutApplyElementwise",value:function(t){return this.applyElementwiseInto(t,this.clone())}},{key:"applyElementwiseInto",value:function(t,r){if(this.rows!==r.rows||this.columns!==r.columns)throw new TypeError("Cannot apply "+t.name+" elementwise on a "+this.rows+"x"+this.columns+" matrix into a "+r.rows+"x"+r.columns+" matrix. Matrices must have the same dimensions.");for(var e=this.data,n=r.data,o=n.length,i=0;i<o;i++)n[i]=t(e[i]);return r}},{key:"copyInto",value:function(t){if(this.rows!==t.rows||this.columns!==t.columns)throw new Error("Cannot copy a "+this.rows+"x"+this.columns+" matrix into a "+t.rows+"x"+t.columns+" matrix.");for(var r=this.data,e=t.data,n=e.length,o=0;o<n;o++)e[o]=r[o];return t}},{key:"setToZero",value:function(){for(var t=this.data,r=t.length,e=0;e<r;e++)t[e]=0}},{key:"print",value:function(t){for(var r=Array.from(this.rowMajorOrderEntries()).map((function(r){return r.toFixed(t)})),e=r.map((function(t){return t.length})),n=Math.max.apply(Math,u(e)),o="-".repeat(this.columns*(n+" | ".length)-" | ".length),i=o+"\n",s=0;s<this.rows;s++){for(var a=0;a<this.columns;a++)i+=p(r[s*this.columns+a],n," ")+" | ";i=i.slice(0,-" | ".length),i+="\n"}return i+=o}}]),t}();function p(t,r,e){var n=r-t.length;return n<=0?t:e.repeat(n)+t}var m={offset:0,requiredValue:2051},d={offset:4},v={offset:8},y={offset:12},w={offset:0,requiredValue:2049},g={offset:4};function b(t,r){var e=function(t){var r=new Uint8Array(t);!function(t){var r=x(t,m.offset),e=m.requiredValue;if(r!==e)throw new Error("The first 4 bytes of an idx3 file must be 0x"+e.toString(16)+", but the first 4 bytes of the provided file were 0x"+r.toString(16))}(r);var e=x(r,d.offset),n=x(r,v.offset),o=x(r,y.offset),i=n*o,s=0,a=new Array(e),u=y.offset+4;for(;s<e;){for(var h=new Array(i),c=0;c<i;c++)h[c]=r[u+s*i+c]/255;a[s]={rows:n,columns:o,matrix:l.columnVector(h)},s++}return a}(t),n=function(t){var r=new Uint8Array(t);!function(t){var r=x(t,w.offset),e=w.requiredValue;if(r!==e)throw new Error("The first 4 bytes of an idx1 file must be 0x"+e.toString(16)+", but the first 4 bytes of the provided file were 0x"+r.toString(16))}(r);for(var e=x(r,g.offset),n=new Array(e),o=g.offset+4,i=0;i<e;i++)n[i]=r[o+i];return n}(r);if(e.length!==n.length)throw new Error("There are "+e.length+" images, but "+n.length+" labels. There must be the same amount of images and labels.");for(var o=new Array(e.length),i=0;i<e.length;i++){var s=e[i],a=s.rows,u=s.columns,h=s.matrix;o[i]={rows:a,columns:u,inputs:h,label:n[i]}}return o}function x(t,r){return t[r]<<24|t[r+1]<<16|t[r+2]<<8|t[r+3]}function O(t){var r=new Array(10).fill(0);r[t.label]=1;var e=l.columnVector(r);return{rows:t.rows,columns:t.columns,inputs:t.inputs,outputs:e}}function _(t){return{rows:t.rows,columns:t.columns,inputs:t.inputs,label:k(t)}}function k(t){for(var r=t.outputs.rowMajorOrderEntries(),e=0;e<r.length;e++)if(1===r[e])return e;throw new Error("A VectorLabeledImage has an output vector without a 1.")}function E(t,r){return function(t){if(Array.isArray(t))return t}(t)||function(t,r){if("undefined"!==typeof Symbol&&Symbol.iterator in Object(t)){var e=[],n=!0,o=!1,i=void 0;try{for(var s,a=t[Symbol.iterator]();!(n=(s=a.next()).done)&&(e.push(s.value),!r||e.length!==r);n=!0);}catch(u){o=!0,i=u}finally{try{n||null==a.return||a.return()}finally{if(o)throw i}}return e}}(t,r)||a(t,r)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}var j=e(0),A=e.n(j),S=self.location,T=A.a.resolve(S.href,"./assets/train60k-images-idx3-ubyte"),I=A.a.resolve(S.href,"./assets/train60k-labels-idx1-ubyte"),M=A.a.resolve(S.href,"./assets/test10k-images-idx3-ubyte"),C=A.a.resolve(S.href,"./assets/test10k-labels-idx1-ubyte"),P=U(T),G=U(I),q=U(M),R=U(C);function U(t){return fetch(t).then((function(r){return 200<=r.status&&r.status<=299?r.arrayBuffer():function(t){var r=t.status,e=t.statusText;return t.text().then((function(t){return r+" ("+e+"): "+t}))}(r).then((function(r){return Promise.reject(new Error("Tried to fetch "+t+" but got the following error: "+r))}))}))}var z,N=Promise.all([P,G,q,R]).then((function(t){var r=E(t,4),e=r[0],n=r[1],o=r[2],i=r[3];return{training:b(e,n).map(O),test:b(o,i)}}));function L(t){if("undefined"===typeof Symbol||null==t[Symbol.iterator]){if(Array.isArray(t)||(t=a(t))){var r=0,e=function(){};return{s:e,n:function(){return r>=t.length?{done:!0}:{done:!1,value:t[r++]}},e:function(t){throw t},f:e}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}var n,o,i=!0,s=!1;return{s:function(){n=t[Symbol.iterator]()},n:function(){var t=n.next();return i=t.done,t},e:function(t){s=!0,o=t},f:function(){try{i||null==n.return||n.return()}finally{if(s)throw o}}}}function F(){return 2*Math.random()-1}function B(){for(var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,r=arguments.length>1&&void 0!==arguments[1]?arguments[1]:1,e=0,n=0;0===e;)e=Math.random();for(;0===n;)n=Math.random();var o=Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*n);return r*o+t}function W(t,r){!function(t){for(var r=0;r<512;r++)for(var e=t.length-1;e>=1;e--){var n=(i=e+1,Math.floor(Math.random()*i)),o=t[e];t[e]=t[n],t[n]=o}var i}(t);for(var e=[],n=0;n<t.length;n+=r)e.push(t.slice(n,n+r));return e}function D(t){for(var r=0,e=t[r],n=1;n<t.length;n++){var o=t[n];o>e&&(e=o,r=n)}return r}!function(t){t.Uniform="Uniform",t.LargeGaussian="LargeGaussian",t.SmallGaussian="SmallGaussian"}(z||(z={}));var V=function(){function t(r,e,n){h(this,t),this.numberOfLayers=void 0,this.weights=void 0,this.biases=void 0,this.log=void 0,this.temp_totalWeightGradients=void 0,this.temp_totalBiasGradients=void 0,this.temp_weightedSums=void 0,this.temp_activations=void 0,this.temp_errors=void 0,this.temp_weightGradients=void 0,this.temp_biasGradients=void 0,this.temp_transposedActivations=void 0,this.temp_weightCosts=void 0,this.temp_transposedWeights=void 0,this.temp_sigmaPrimeOfWeightedSums=void 0,this.layerSizes=void 0;for(var o=[r[1].columns],i=1;i<r.length;i++)o.push(r[i].rows);this.layerSizes=o,this.numberOfLayers=o.length,this.weights=r,this.biases=e,this.log=n||function(){},this.temp_totalWeightGradients=$(r),this.temp_totalBiasGradients=$(e);for(var s=[],a=[l.zeros(r[1].columns,1)],u=1;u<this.numberOfLayers;u++)s[u]=l.zeros(r[u].rows,1),a[u]=l.zeros(r[u].rows,1);this.temp_weightedSums=s,this.temp_activations=a,this.temp_errors=$(this.temp_weightedSums),this.temp_weightGradients=$(r),this.temp_biasGradients=$(e);for(var c=this.temp_activations,f=new Array(c.length),p=0;p<c.length;p++)f[p]=l.zeros(c[p].columns,c[p].rows);this.temp_transposedActivations=f,this.temp_weightCosts=$(this.temp_weightGradients);for(var m=this.weights,d=new Array(m.length),v=1;v<m.length;v++)d[v]=l.zeros(m[v].columns,m[v].rows);this.temp_transposedWeights=d,this.temp_sigmaPrimeOfWeightedSums=$(this.temp_weightedSums)}return f(t,null,[{key:"fromWeightsAndBiases",value:function(r,e){return new t(r,e)}},{key:"fromLayerSizes",value:function(r,e,n){for(var o=r.length,i=new Array(o),s=new Array(o),a=1;a<o;a++){var u=a-1,h=r[a],c=r[u];i[a]=l.zeros(h,c),s[a]=l.zeros(h,1)}return function(t,r){for(var e=function(e){var n=r[e],o=function(){switch(t){case z.Uniform:return F;case z.LargeGaussian:return function(){return B(0,1)};case z.SmallGaussian:return function(){return B(0,1/Math.sqrt(n.columns))}}}();n.applyElementwiseInto(o,n)},n=1;n<r.length;n++)e(n)}(e,i),new t(i,s,n)}}]),f(t,[{key:"stochasticGradientDescent",value:function(t,r,e){for(var n=r.batchSize,o=r.epochs,i=r.learningRate,s=t.length,a=0;a<o;a++){var u,h=L(W(t,n));try{for(h.s();!(u=h.n()).done;)for(var c=u.value,f=this.getTotalGradients(c,r.regularizationRate,s),l=f.weightGradients,p=f.biasGradients,m=1;m<this.numberOfLayers;m++)l[m].mutMultiplyScalar(i/c.length),p[m].mutMultiplyScalar(i/c.length),this.weights[m].mutSubtract(l[m]),this.biases[m].mutSubtract(p[m])}catch(v){h.e(v)}finally{h.f()}if(void 0!==e){var d=this.test(e);this.log(d,a)}}}},{key:"getTotalGradients",value:function(t,r,e){var n,o=this.resetTotalGradientTemps(),i=o.weightGradients,s=o.biasGradients,a=L(t);try{for(a.s();!(n=a.n()).done;)for(var u=n.value,h=this.getGradients(u,r,e),c=h.weightGradients,f=h.biasGradients,l=1;l<this.numberOfLayers;l++)i[l].mutAdd(c[l]),s[l].mutAdd(f[l])}catch(p){a.e(p)}finally{a.f()}return{weightGradients:i,biasGradients:s}}},{key:"resetTotalGradientTemps",value:function(){for(var t=this.layerSizes.length,r=this.temp_totalWeightGradients,e=this.temp_totalBiasGradients,n=1;n<t;n++)r[n].setToZero(),e[n].setToZero();return{weightGradients:r,biasGradients:e}}},{key:"getGradients",value:function(t,r,e){var n=this.numberOfLayers,o=this.performForwardPass(t.inputs),i=o.weightedSums,s=o.activations,a=this.temp_errors,u=this.temp_weightGradients,h=this.temp_biasGradients,c=s[this.numberOfLayers-1].subtractInto(t.outputs,a[n-1]);c.multiplyInto(s[n-2].transposeInto(this.temp_transposedActivations[n-2]),u[n-1]).mutAdd(this.weights[n-1].multiplyScalarInto(r/e,this.temp_weightCosts[n-1])),c.copyInto(h[n-1]);for(var f=this.numberOfLayers-2;f>=1;f--){var l=this.weights[f+1].transposeInto(this.temp_transposedWeights[f+1]).multiplyInto(a[f+1],a[f]).mutHadamard(i[f].applyElementwiseInto(Z,this.temp_sigmaPrimeOfWeightedSums[f]));l.multiplyInto(s[f-1].transposeInto(this.temp_transposedActivations[f-1]),u[f]).mutAdd(this.weights[f].multiplyScalarInto(r/e,this.temp_weightCosts[f])),l.copyInto(h[f])}return{weightGradients:u,biasGradients:h}}},{key:"performForwardPass",value:function(t){var r=this.temp_weightedSums,e=this.temp_activations;e[0]=t;for(var n=1;n<this.numberOfLayers;n++){var o=n-1;this.weights[n].multiplyInto(e[o],r[n]).mutAdd(this.biases[n]).applyElementwiseInto(H,e[n])}return{weightedSums:r,activations:e}}},{key:"test",value:function(t){var r,e=0,n=L(t);try{for(n.s();!(r=n.n()).done;){var o=r.value;D(this.performForwardPass(o.inputs).activations[this.numberOfLayers-1].rowMajorOrderEntries())===o.label&&e++}}catch(i){n.e(i)}finally{n.f()}return{correct:e,total:t.length}}},{key:"getWeights",value:function(){return this.weights}},{key:"getBiases",value:function(){return this.biases}}]),t}();function H(t){return 1/(1+Math.exp(-t))}function Z(t){var r=H(t);return r*(1-r)}function $(t){for(var r=[],e=1;e<t.length;e++){var n=t[e];r[e]=l.zeros(n.rows,n.columns)}return r}var Y,K=function(t,r){return V.fromWeightsAndBiases(t,r)};function J(t){var r=function(t){for(var r=t.getWeights(),e=t.getBiases(),n=0,o=1;o<r.length;o++){var i=r[o],s=i.rows*i.columns;n+=s;var a=e[o],u=a.rows*a.columns;n+=u}for(var h=new Float64Array(n),c=0,f=1;f<r.length;f++){var l=r[f].rowMajorOrderEntries();h.set(l,c),c+=l.length;var p=e[f].rowMajorOrderEntries();h.set(p,c),c+=p.length}return h}(t),e=(1+t.layerSizes.length)*Uint32Array.BYTES_PER_ELEMENT,n=new ArrayBuffer(e+r.length*r.BYTES_PER_ELEMENT),o=new Uint32Array(n,0,e/Uint32Array.BYTES_PER_ELEMENT);o[0]=t.layerSizes.length;for(var i=0;i<t.layerSizes.length;i++)o[1+i]=t.layerSizes[i];return new Float64Array(n,e).set(r),n}function Q(t){for(var r=new Array(t.length),e=0;e<t.length;e++)r[e]=t[e];return r}!function(t){t[t.StartTrainingRequest=0]="StartTrainingRequest",t[t.TrainingEpochCompleteNotification=1]="TrainingEpochCompleteNotification",t[t.TerminateTrainingRequest=2]="TerminateTrainingRequest",t[t.TerminateTrainingResponse=3]="TerminateTrainingResponse",t[t.StartTestingRequest=4]="StartTestingRequest",t[t.TestCompleteNotification=5]="TestCompleteNotification"}(Y||(Y={}));var X=e(1),tt=!1,rt=X.option.none();function et(t){var r={messageType:Y.TerminateTrainingResponse,networkBuffer:J(t)};self.postMessage(r,[r.networkBuffer])}self.addEventListener("message",(function(t){var r=t.data;if(null!==r&&"object"===typeof r&&"messageType"in r){var e=r;switch(e.messageType){case Y.StartTrainingRequest:!function(t){var r=function(t){for(var r=new Uint32Array(t,0,1)[0],e=new Uint32Array(t,4,r),n=new Float64Array(t.slice(Uint32Array.BYTES_PER_ELEMENT*(1+r))),o=[],i=[],s=0,a=1;a<e.length;a++){var u=e[a],h=u,c=e[a-1],f=h*c;o[a]=l.fromRowMajorOrderEntries(h,c,Q(n.subarray(s,s+f))),s+=f;var p=u;i[a]=l.fromRowMajorOrderEntries(p,1,Q(n.subarray(s,s+p))),s+=p}return K(o,i)}(t.networkBuffer);N.then((function(e){!function(t,r,e,n){var o=0;function s(){tt?et(t):requestAnimationFrame(a)}function a(){t.stochasticGradientDescent(e,i(i({},r),{},{epochs:1})),function(t,r){var e={messageType:Y.TrainingEpochCompleteNotification,accuracyRate:t,epoch:r};self.postMessage(e)}(t.test(n),o),++o<r.epochs?s():rt=X.option.some(t)}s()}(r,t.hyperParams,e.training.slice(0,5e4),e.training.slice(5e4).map(_))}))}(e);break;case Y.TerminateTrainingRequest:rt.match({none:function(){tt=!0},some:et});break;default:}}}))}]);
//# sourceMappingURL=d93723539f32b8051de9.worker.js.map